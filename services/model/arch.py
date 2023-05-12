import dataclasses
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from typing import Union

import numpy as np
import datetime
import pandas as pd
import torch
from datasets import Dataset
from sqlalchemy import create_engine
from sqlalchemy import func
from sqlalchemy import select
from tqdm import tqdm

from transformers import T5ForConditionalGeneration, T5Tokenizer, EvalPrediction
from transformers import (
  HfArgumentParser,
  DataCollator,
  Trainer,
  TrainingArguments,
  set_seed,
)
from transformers import TrainerCallback

from services.config import model_path
from services.config import predict_result_path
from services.const import ModelStatus
from services.model import device
from services.model.util.model_input import prediction_model_preprocessing
from services.model.util.model_input import rationale_model_preprocessing
from services.orm.tables import label_result
from services.orm.tables import model_info


logger = logging.getLogger(__name__)


class MyCallback(TrainerCallback):
  "A callback that prints a message at the beginning of training"
  model_res = dict()
  engine = create_engine("sqlite:///test.db", echo=True)
  @property
  def model_res_info(self):
    if not self.model_res:
      sql = (select(model_info.c.id,
                    model_info.c.model_uuid,
                    model_info.c.label_id,
                    model_info.c.extra,
                    model_info.c.iteration)
             .where(model_info.c.model_uuid == self.model_id))
      with self.engine.connect() as conn:
        res = conn.execute(sql).fetchone()._asdict()
      if res:
        self.model_res = res
    return self.model_res

  def __init__(self, model_id: str, total_steps: int = 1, current_step: int = 1):
    print('#####', model_id, total_steps, current_step, '#####')
    self.model_id = model_id
    self.total_steps = total_steps
    self.current_step = current_step

  def on_train_begin(self, args, state, control, **kwargs):
    print('#####', self.model_id, self.total_steps, self.current_step, '#####')
    if self.model_res_info:
      sql = (model_info
             .update()
             .where(model_info.c.model_uuid == self.model_id)
             .values({'extra': func.json_patch(model_info.c.extra,
                                               json.dumps({'train_begin': True,
                                                           'total_steps': self.total_steps,
                                                           'current_step': self.current_step,
                                                           'train_end': False,
                                                           'begin_time': time.time()})),
                      'status': 1}))
      with self.engine.begin() as conn:
        conn.execute(sql)
    print("######### Training begin ###########")

  def on_epoch_end(self, args, state, control, **kwargs):
    if self.model_res_info:
      sql = (model_info
             .update()
             .where(model_info.c.model_uuid == self.model_id)
             .values({'extra': func.json_set(model_info.c.extra,
                                             '$.progress',
                                             f'{state.epoch}/{state.num_train_epochs}')}))
      with self.engine.begin() as conn:
        conn.execute(sql)
    print(state.epoch, '#####', state.num_train_epochs)

  def on_train_end(self, args, state, control, **kwargs):
    if self.model_res_info:
      train_info = {
        'current_step': self.current_step
      }
      status = 1
      if self.current_step == self.total_steps:
        train_info.update({'train_end': True,
                           'train_begin': False,
                           'end_time': datetime.datetime.utcnow().isoformat()})
        status = 0
      sql = (model_info
             .update()
             .where(model_info.c.model_uuid == self.model_id)
             .values({'extra': func.json_patch(model_info.c.extra,
                                               json.dumps(train_info)),
                      'status': status}))
      with self.engine.begin() as conn:
        conn.execute(sql)
    print("######### Training End ###########")

# prepares lm_labels from target_ids, returns examples with keys as expected by the forward method
# this is necessacry because the trainer directly passes this dict as arguments to the model
# so make sure the keys match the parameter names of the forward method
@dataclass
class T2TDataCollator():
  def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
    """
    Take a list of samples from a Dataset and collate them into a batch.
    Returns:
        A dictionary of tensors
    """

    input_ids = torch.stack([example['input_ids'] for example in batch])
    labels = torch.stack([example['target_ids'] for example in batch])
    labels[labels[:, :] == 0] = -100
    attention_mask = torch.stack([example['attention_mask'] for example in batch])
    decoder_attention_mask = torch.stack([example['target_attention_mask'] for example in batch])

    return {
      'input_ids': input_ids,
      'attention_mask': attention_mask,
      'labels': labels,
      'decoder_attention_mask': decoder_attention_mask
    }


@dataclass
class ModelArguments:
  """
  Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
  """

  model_name_or_path: str = field(
      metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
  )
  tokenizer_name: Optional[str] = field(
      default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
  )
  cache_dir: Optional[str] = field(
      default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
  )


@dataclass
class DataTrainingArguments:
  """
  Arguments pertaining to what data we are going to input our model for training and eval.
  """
  train_file_path: Optional[str] = field(
      default='train_data.pt',
      metadata={"help": "Path for cached train dataset"},
  )
  valid_file_path: Optional[str] = field(
      default='valid_data.pt',
      metadata={"help": "Path for cached valid dataset"},
  )
  max_len: Optional[int] = field(
      default=512,
      metadata={"help": "Max input length for the source text"},
  )
  target_max_len: Optional[int] = field(
      default=32,
      metadata={"help": "Max input length for the target text"},
  )


def finetune(config_json, model_id: str = None, total_steps: int = 1, current_step: int = 1):
  # See all possible arguments in src/transformers/training_args.py
  # or by passing the --help flag to this script.
  # We now keep distinct sets of args, for a cleaner separation of concerns.

  parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

  # we will load the arguments from a json file,
  # make sure you save the arguments in at ./args.json
  model_args, data_args, training_args = parser.parse_dict(config_json)

  if (
      os.path.exists(training_args.output_dir)
      and os.listdir(training_args.output_dir)
      and training_args.do_train
      and not training_args.overwrite_output_dir
  ):
    raise ValueError(
        f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
    )

  # Setup logging
  # logging.basicConfig(
  #     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
  #     datefmt="%m/%d/%Y %H:%M:%S",
  #     level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
  # )
  logger.warning(
      "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
      training_args.local_rank,
      training_args.device,
      training_args.n_gpu,
      bool(training_args.local_rank != -1),
      training_args.fp16,
  )
  logger.info("Training/evaluation parameters %s", training_args)

  # Set seed
  set_seed(training_args.seed)

  # Load pretrained model and tokenizer
  #
  # Distributed training:
  # The .from_pretrained methods guarantee that only one local process can concurrently
  # download model & vocab.

  tokenizer = T5Tokenizer.from_pretrained(
      model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
      # cache_dir=model_args.cache_dir,
  )
  tokenizer.add_special_tokens({'sep_token': '<sep>'})

  model = T5ForConditionalGeneration.from_pretrained(
      model_args.model_name_or_path,
      # cache_dir=model_args.cache_dir,
  )

  # Get datasets
  print('loading data')
  train_dataset = torch.load(data_args.train_file_path)
  valid_dataset = torch.load(data_args.valid_file_path)
  print('loading done')

  # Initialize our Trainer
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=valid_dataset,
      data_collator=T2TDataCollator(),
      callbacks=[MyCallback(model_id=model_id, total_steps=total_steps, current_step=current_step)]
  )

  # Training
  if training_args.do_train:
    trainer.train(
        model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
    )
    trainer.save_model()
    # state_dict = trainer.model.state_dict()
    # torch.save(state_dict, training_args.output_dir)

    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_process_zero():
      tokenizer.save_pretrained(training_args.output_dir)

  # Evaluation
  results = {}
  if training_args.do_eval and training_args.local_rank in [-1, 0]:
    logger.info("*** Evaluate ***")

    eval_output = trainer.evaluate()

    output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
      logger.info("***** Eval results *****")
      for key in sorted(eval_output.keys()):
        logger.info("  %s = %s", key, str(eval_output[key]))
        writer.write("%s = %s\n" % (key, str(eval_output[key])))

    results.update(eval_output)

  return results


def predict(test_dataset, model_path):
  model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
  tokenizer = T5Tokenizer.from_pretrained(model_path)

  dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

  answers = []
  for batch in tqdm(dataloader):
    outs = model.generate(input_ids=batch['input_ids'].to(device),
                          attention_mask=batch['attention_mask'].to(device),
                          max_length=64,
                          early_stopping=True)
    outs = [tokenizer.decode(ids, skip_special_tokens=True).encode('utf-8').decode('utf-8') for ids in outs]
    answers.extend(outs)

  # references = []
  # for ref, pred in zip(test_dataset, answers):
  #   predictions.append(pred)
  #   # references.append(ref['answer'])

  #   references.append(tokenizer.decode(ref['target_ids'], skip_special_tokens=True))

  # print("1st predicted:", predictions[0])
  # print("1st groundtruth:", references[0])
  # assert len(predictions) == len(references)

  return answers


def predict_pipeline(data_predict: Union[list, dict, pd.DataFrame],
                     model_id: str,
                     label_id: int,
                     labels: list = None,
                     column_1: str = 'premise',
                     column_2: str = 'hypothesis',
                     explanation_column: str = 'explanation_1'):

  result_path = os.path.join(predict_result_path, str(label_id))
  os.makedirs(result_path, exist_ok=True)
  with open(result_path+f'/{model_id}.json', 'w') as f:
    json.dump({'status': 'predicting'}, f)

  labels = labels or ['entailment', 'neutral', 'contradiction']

  if isinstance(data_predict, pd.DataFrame):
    predict_dataset = Dataset.from_pandas(data_predict)
  elif isinstance(data_predict, dict):
    predict_dataset = Dataset.from_dict(data_predict)
  else:
    predict_dataset = Dataset.from_list(data_predict)

  rationale_model_predict_dataset = rationale_model_preprocessing(predict_dataset,
                                                                  labels=labels,
                                                                  column_1=column_1,
                                                                  column_2=column_2,
                                                                  explanation_column=explanation_column)

  work_dir = os.path.join(model_path, model_id)
  r_model_path = os.path.join(work_dir, 'model', 'rationale')
  p_model_path = os.path.join(work_dir, 'model', 'prediction')

  predicted_rationale = predict(rationale_model_predict_dataset, r_model_path)

  ## preprocess generated rationales
  predict_dataset = predict_dataset.add_column("generated_rationale",
                                               predicted_rationale)

  prediction_model_batch_train_dataset = prediction_model_preprocessing(predict_dataset,
                                                                        labels=labels,
                                                                        column_1=column_1,
                                                                        column_2=column_2,
                                                                        explanation_column='generated_rationale')
  predicted_label = predict(prediction_model_batch_train_dataset, p_model_path)

  with open(result_path+f'/{model_id}.json', 'w') as f:
    json.dump({'status': 'finished',
               'predicted_rationale': predicted_rationale,
               'predicted_label': predicted_label},
              f)

  return predicted_rationale, predicted_label

