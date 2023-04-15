# -*- coding: utf-8 -*-
"""AL_iteration.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sBjYQ_AFIsfaICvd5FxRiLRXs4j7rPbl
"""
import json
import logging
import os
import os.path
import time
from dataclasses import dataclass
from dataclasses import field
from functools import partial
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import pandas as pd
import torch
from datasets import Dataset
from datasets import concatenate_datasets
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tqdm import tqdm
from transformers import HfArgumentParser
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
from transformers import Trainer
from transformers import TrainingArguments
from transformers import set_seed


rationale_model_path = 'model_data/r_model'
prediction_model_path = 'model_data/p_model'

rationale_model_data_path = 'model_data/r_model_data'
prediction_model_data_path = 'model_data/p_model_data'

predict_batch_size = 192

logger1 = logging.getLogger(__name__)

handler1 = logging.StreamHandler()
handler2 = logging.FileHandler(filename="run1.log")

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler2.setLevel(logging.DEBUG)
handler2.setFormatter(formatter)

logger1.addHandler(handler1)
logger1.addHandler(handler2)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Device: ", device)

tokenizer = T5Tokenizer.from_pretrained('t5-base')
tokenizer.add_special_tokens({'sep_token': '<sep>'})
model_SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")


test_data = {'premise': ['A person on a horse jumps over a broken down airplane.',
                         'A person on a horse jumps over a broken down airplane.',
                         'A person on a horse jumps over a broken down airplane.',
                         'Children smiling and waving at camera',
                         'Children smiling and waving at camera',
                         'Children smiling and waving at camera',
                         'A boy is jumping on skateboard in the middle of a red bridge.',
                         'A boy is jumping on skateboard in the middle of a red bridge.',
                         'A boy is jumping on skateboard in the middle of a red bridge.',
                         'An older man sits with his orange juice at a small table in a coffee shop while employees in bright colored shirts smile in the background.'],
             'hypothesis': ['A person is training his horse for a competition.',
                            'A person is at a diner, ordering an omelette.',
                            'A person is outdoors, on a horse.',
                            'They are smiling at their parents',
                            'There are children present',
                            'The kids are frowning',
                            'The boy skates down the sidewalk.',
                            'The boy does a skateboarding trick.',
                            'The boy is wearing safety equipment.',
                            'An older man drinks his juice as he waits for his daughter to get off work.'],
             'label': [1, 2, 0, 1, 0, 2, 2, 0, 1, 1],
             'explanation_1': ['the person is not necessarily training his horse',
                               'One cannot be on a jumping horse cannot be a diner ordering food.',
                               'a broken down airplane is outdoors',
                               'Just because they are smiling and waving at a camera does not imply their parents or anyone is anyone behind it',
                               'The children must be present to see them smiling and waving.',
                               'One cannot be smiling and frowning at the same time.',
                               'One cannot be in the middle of a bridge if they are on the sidewalk.',
                               'jumping on skateboard is the same as doing trick on skateboard.',
                               'Just because the boy is jumping on a skateboard does not imply he is wearing safety equipment',
                               'it is not necessarily true the man drinks his juice']}

labels = ['entailment', 'neutral', 'contradiction']


def rationale_model_define_input_1(example,
                                   label: list,
                                   column_1: str,
                                   column_2: str,
                                   explanation_column: str):
  label_content = ""
  for i in range(len(label)):
    label_content += 'choice' + str(i + 1) + ': ' + str(label[i]) + ' '

  example['input_text'] = ('explain: what is the relationship between %s and %s ' % (
  example[column_1], example[column_2])) + label_content
  example['target_text'] = '%s' % example[explanation_column]

  return example


def prediction_model_define_input_1(example,
                                    label: list,
                                    column_1: str,
                                    column_2: str,
                                    explanation_column: str = 'generated_rationale',
                                    label_column: str = 'label'):
  label_content = ""
  for i in range(len(label)):
    label_content += 'choice' + str(i + 1) + ': ' + str(label[i]) + ' '

  example['input_text'] = ('question: what is the relationship between %s and %s ' % (
  example[column_1], example[column_2])) + label_content + (
                                ' <sep> because %s' % (example[explanation_column]))
  example['target_text'] = '%s' % label[int(example[label_column])]

  return example


def rationale_model_define_input_2(example, label: list):
  label_content = ""
  for i in range(len(label)):
    label_content += 'choice' + str(i + 1) + ': ' + str(label[i]) + ' '

  example['input_text'] = ('explain: what is the relationship between %s and %s ' % (
  example['sentence'], example['label'])) + label_content
  example['target_text'] = '%s' % example['explanation']

  return example


def prediction_model_define_input_2(example, label: list):
  label_content = ""
  for i in range(len(label)):
    label_content += 'choice' + str(i + 1) + ': ' + str(label[i]) + ' '

  example['input_text'] = ('question: what is the relationship between %s and %s ' % (
  example['hypothesis'], example['premise'])) + label_content + (
                                ' <sep> because %s' % (example['generated_rationale']))
  example['target_text'] = '%s' % label[int(example['label'])]

  return example


def convert_to_features(example_batch):
  input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'],
                                                add_special_tokens=True,
                                                truncation=True,
                                                pad_to_max_length=True,
                                                max_length=512)
  target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'],
                                                 add_special_tokens=True,
                                                 truncation=True,
                                                 pad_to_max_length=True,
                                                 max_length=64)

  encodings = {
    'input_ids': input_encodings['input_ids'],
    'attention_mask': input_encodings['attention_mask'],
    'target_ids': target_encodings['input_ids'],
    'target_attention_mask': target_encodings['attention_mask']
  }

  return encodings


def rationale_model_preprocessing(input_dataset,
                                  labels,
                                  column_1: str,
                                  column_2: str,
                                  explanation_column: str):
  rationale_model_inout = partial(rationale_model_define_input_1,
                                  label=labels,
                                  column_1=column_1,
                                  column_2=column_2,
                                  explanation_column=explanation_column)
  input_dataset = input_dataset.map(rationale_model_inout, load_from_cache_file=False)

  input_dataset = input_dataset.map(convert_to_features, batched=True, load_from_cache_file=False)

  columns = ['input_ids', 'target_ids', 'attention_mask', 'target_attention_mask']
  input_dataset.set_format(type='torch', columns=columns)
  return input_dataset


def prediction_model_preprocessing(input_dataset,
                                   labels,
                                   column_1: str,
                                   column_2: str,
                                   explanation_column: str,
                                   label_column: str):
  prediction_model_input = partial(prediction_model_define_input_1,
                                   label=labels,
                                   column_1=column_1,
                                   column_2=column_2,
                                   explanation_column=explanation_column,
                                   label_column=label_column)
  input_dataset = input_dataset.map(prediction_model_input, load_from_cache_file=False)
  input_dataset = input_dataset.map(convert_to_features, batched=True, load_from_cache_file=False)

  columns = ['input_ids', 'target_ids', 'attention_mask', 'target_attention_mask']
  input_dataset.set_format(type='torch', columns=columns)
  return input_dataset


def trans_data_to_dataset(raw_data) -> Dataset:
  if isinstance(raw_data, pd.DataFrame):
    trans_dataset = Dataset.from_pandas(raw_data)
  elif isinstance(raw_data, dict):
    trans_dataset = Dataset.from_dict(raw_data)
  elif isinstance(raw_data, list):
    trans_dataset = Dataset.from_list(raw_data)
  else:
    trans_dataset = raw_data
  return trans_dataset


def similarity_batch_selection(train_data,
                               previous_batch_train_data,
                               criteria: str,
                               current_iter: int = 0,
                               num_batch: int = 20,
                               column_1: str = 'premise',
                               column_2: str = 'hypothesis',
                               explanation_column: str = 'explanation_1'):
  logger1.warning(f'start select data by similarity, to_select: {len(train_data)}, prev_selected: {len(previous_batch_train_data)}')
  train_dataset = trans_data_to_dataset(train_data)
  previous_batch_train_dataset = trans_data_to_dataset(previous_batch_train_data)

  ## THESE TWO METHODS ONLY COMPARE THE TASK CONTENT
  if criteria in ['combined', 'even']:

    ## During the first iteration, there's no human annotattion
    ## So we compare the similarity between original content
    if current_iter == 0:
      sentence_list = [' '.join([i[column_2], i[column_1]]) for i in train_dataset]
      rationale_list = [' '.join([i[column_2], i[column_1]]) for i in train_dataset]
    else:
      sentence_list = [' '.join([i[column_2], i[column_1]]) for i in train_dataset]
      rationale_list = [' '.join([i[column_2], i[column_1]]) for i in previous_batch_train_dataset]

    sentence_list_embeddings = model_SentenceTransformer.encode(sentence_list, convert_to_tensor=True).to(
      device)
    rationale_list_embeddings = model_SentenceTransformer.encode(rationale_list, convert_to_tensor=True).to(
      device)

    cosine_scores = util.cos_sim(sentence_list_embeddings, rationale_list_embeddings)

    mean_cosine_scores = torch.mean(cosine_scores, 1, True)
    mean_cosine_scores_list = [i.item() for i in mean_cosine_scores]
    mean_cosine_scores_list_sorted_with_idx = sorted(range(len(mean_cosine_scores_list)),
                                                     key=lambda i: mean_cosine_scores_list[i])

    # combine half top ranked ones and half bottom ranked ones
    if criteria == 'combined':
      sampled_idx = mean_cosine_scores_list_sorted_with_idx[
                    -int(num_batch / 2):] + mean_cosine_scores_list_sorted_with_idx[:int(num_batch / 2)]
    # evenly select examples from ranked data
    elif criteria == 'even':
      step = int(len(train_dataset) / num_batch)
      sampled_idx = mean_cosine_scores_list_sorted_with_idx[0::step]

  ## THESE TWO METHODS COMPARE THE RATIONALES AND TASK CONTENT
  elif criteria in ['combined_rationale', 'even_rationale']:

    ## During the first iteration, there's no human annotattion
    ## So we compare the similarity between original content
    if current_iter == 0:
      sentence_list = [' '.join([i[column_2], i[column_1]]) for i in train_dataset]
      rationale_list = [' '.join([i[column_2], i[column_1]]) for i in train_dataset]
    else:
      sentence_list = [' '.join([i[column_2], i[column_1]]) for i in train_dataset]
      rationale_list = [i[explanation_column] for i in previous_batch_train_dataset]

    sentence_list_embeddings = model_SentenceTransformer.encode(sentence_list, convert_to_tensor=True).to(
      device)
    rationale_list_embeddings = model_SentenceTransformer.encode(rationale_list, convert_to_tensor=True).to(
      device)

    cosine_scores = util.cos_sim(sentence_list_embeddings, rationale_list_embeddings)

    mean_cosine_scores = torch.mean(cosine_scores, 1, True)
    mean_cosine_scores_list = [i.item() for i in mean_cosine_scores]
    mean_cosine_scores_list_sorted_with_idx = sorted(range(len(mean_cosine_scores_list)),
                                                     key=lambda i: mean_cosine_scores_list[i])

    if criteria == 'combined_rationale':
      sampled_idx = mean_cosine_scores_list_sorted_with_idx[
                    -int(num_batch / 2):] + mean_cosine_scores_list_sorted_with_idx[:int(num_batch / 2)]
    elif criteria == 'even_rationale':
      step = int(len(train_dataset) / num_batch)
      sampled_idx = mean_cosine_scores_list_sorted_with_idx[0::step]

  ## finish sampled index selection
  remaining_idx = list(range(len(train_dataset)))
  for i in sampled_idx:
    remaining_idx.remove(i)

  # batch_dataset = train_dataset.select(sampled_idx)
  # remain_dataset = train_dataset.select(remaining_idx)
  # return batch_dataset, remain_dataset
  logger1.warning(f'End select data by similarity, selected: {len(sampled_idx)}, remain: {len(remaining_idx)}')
  return train_dataset.select(sampled_idx), train_dataset.select(remaining_idx)


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


def predict(test_dataset, model_name, model_path, flag, iteration):
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

  predictions = []
  references = []
  for ref, pred in zip(test_dataset, answers):
    predictions.append(pred)
    # references.append(ref['answer'])

    references.append(tokenizer.decode(ref['target_ids'], skip_special_tokens=True))

  print("1st predicted:", predictions[0])
  print("1st groundtruth:", references[0])
  assert len(predictions) == len(references)
  # print(len(predictions), len(references))

  return predictions


def finetune(config_json):
  parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
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
  #     format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
  #     datefmt="%m/%d/%Y %H:%M:%S",
  #     level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
  # )
  logger1.warning(
      "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
      training_args.local_rank,
      training_args.device,
      training_args.n_gpu,
      bool(training_args.local_rank != -1),
      training_args.fp16,
  )
  # logger.info("Training/evaluation parameters %s", training_args)
  set_seed(training_args.seed)

  tokenizer = T5Tokenizer.from_pretrained(
      model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
  )
  tokenizer.add_special_tokens({'sep_token': '<sep>'})

  model = T5ForConditionalGeneration.from_pretrained(
      model_args.model_name_or_path,
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
      data_collator=T2TDataCollator()
  )

  # Training
  if training_args.do_train:
    trainer.train(
        model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
    )
    trainer.save_model()

    if trainer.is_world_process_zero():
      tokenizer.save_pretrained(training_args.output_dir)

  # Evaluation
  results = {}
  if training_args.do_eval and training_args.local_rank in [-1, 0]:
    logger1.info("*** Evaluate ***")

    eval_output = trainer.evaluate()

    output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
      logger1.info("***** Eval results *****")
      for key in sorted(eval_output.keys()):
        logger1.info("  %s = %s", key, str(eval_output[key]))
        writer.write("%s = %s\n" % (key, str(eval_output[key])))

    results.update(eval_output)

  return results


def predict_pipeline(data_predict: Union[list, dict, pd.DataFrame],
                     rationale_model_path: str,
                     predict_model_path: str,
                     labels: list = None,
                     column_1: str = 'premise',
                     column_2: str = 'hypothesis',
                     explanation_column: str = 'explanation_1',
                     label_column: str = 'label'):
  def _predict(test_dataset, model_path):
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_path)

    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=predict_batch_size)

    answers = []
    for batch in tqdm(dataloader):
      outs = model.generate(input_ids=batch['input_ids'].to(device),
                            attention_mask=batch['attention_mask'].to(device),
                            max_length=64,
                            early_stopping=True)
      outs = [tokenizer.decode(ids, skip_special_tokens=True).encode('utf-8').decode('utf-8') for ids in outs]
      answers.extend(outs)

    predictions = []
    references = []
    for ref, pred in zip(test_dataset, answers):
      predictions.append(pred)
      # references.append(ref['answer'])

      references.append(tokenizer.decode(ref['target_ids'], skip_special_tokens=True))

    logger1.warning(f"1st predicted:  {predictions[0]}")
    logger1.warning(f"1st groundtruth:  {references[0]}")
    assert len(predictions) == len(references)

    return predictions


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

  predicted_rationale = _predict(rationale_model_predict_dataset, rationale_model_path)

  ## preprocess generated rationales
  predict_dataset = predict_dataset.add_column("generated_rationale",
                                               predicted_rationale)

  prediction_model_batch_train_dataset = prediction_model_preprocessing(predict_dataset,
                                                                        labels=labels,
                                                                        column_1=column_1,
                                                                        column_2=column_2,
                                                                        explanation_column='generated_rationale',
                                                                        label_column=label_column)
  predicted_label = _predict(prediction_model_batch_train_dataset, predict_model_path)
  return predicted_rationale, predicted_label


def one_iter(labeled_data: Union[list, dict],
             column_1: str = 'premise',
             column_2: str = 'hypothesis',
             explanation_column: str = 'explanation_1',
             label_column: str = 'label',
             labels: list = None,
             model_id: str = 'test_model',
             rationale_model: str = rationale_model_path,
             prediction_model: str = prediction_model_path,
             curr_iter: int = 0,
             epochs: int = 10,
             base_model: str = 't5-base'):
  labels = labels or ['entailment', 'neutral', 'contradiction']

  if isinstance(labeled_data, pd.DataFrame):
    train_dataset = Dataset.from_pandas(labeled_data)
  elif isinstance(labeled_data, dict):
    train_dataset = Dataset.from_dict(labeled_data)
  elif isinstance(labeled_data, list):
    train_dataset = Dataset.from_list(labeled_data)
  else:
    train_dataset = labeled_data

  rationale_model_batch_train_dataset = rationale_model_preprocessing(train_dataset,
                                                                      labels=labels,
                                                                      column_1=column_1,
                                                                      column_2=column_2,
                                                                      explanation_column=explanation_column)
  rationale_model_train_file_path = os.path.join(rationale_model_data_path, 'train_data.pt')
  rationale_model_valid_file_path = os.path.join(rationale_model_data_path, 'valid_data.pt')
  torch.save(rationale_model_batch_train_dataset, rationale_model_train_file_path)
  torch.save(rationale_model_batch_train_dataset, rationale_model_valid_file_path)

  ## fine-tune rationale model
  # set config and load model (1st time load pretrained model)

  rationale_model_config_json = {
    "model_name_or_path": base_model if curr_iter == 0 else rationale_model,
    "tokenizer_name": base_model if curr_iter == 0 else rationale_model,
    "max_len": 512,
    "target_max_len": 64,
    "train_file_path": rationale_model_train_file_path,
    "valid_file_path": rationale_model_valid_file_path,
    "output_dir": rationale_model,
    "overwrite_output_dir": True,
    "per_device_train_batch_size": 10,
    "per_device_eval_batch_size": 10,
    "gradient_accumulation_steps": 6,
    "learning_rate": 1e-4,
    "num_train_epochs": epochs,
    "do_train": True,
    "do_eval": False,
    "prediction_loss_only": True,
    "remove_unused_columns": False,
    "save_strategy": 'no',
    "evaluation_strategy": 'no',
    "save_total_limit": 1,
    "load_best_model_at_end": True
  }

  ##print("=== START FINETUNE MODEL RG ===")

  finetune(rationale_model_config_json)

  ##print("=== START GENERATE RATIONALE FOR MODEL P ===")

  predicted_rationale = predict(rationale_model_batch_train_dataset,
                                'rationale',
                                rationale_model,
                                'ignore',
                                curr_iter)

  ## preprocess generated rationales
  prediction_model_batch_train_dataset = train_dataset.add_column("generated_rationale", predicted_rationale)
  print(predicted_rationale)
  prediction_model_batch_train_dataset = prediction_model_preprocessing(prediction_model_batch_train_dataset,
                                                                        labels=labels,
                                                                        column_1=column_1,
                                                                        column_2=column_2,
                                                                        explanation_column=explanation_column,
                                                                        label_column=label_column)
  prediction_model_train_file_path = os.path.join(prediction_model_data_path, 'train_data.pt')
  prediction_model_valid_file_path = os.path.join(prediction_model_data_path, 'valid_data.pt')
  torch.save(prediction_model_batch_train_dataset, prediction_model_train_file_path)
  torch.save(prediction_model_batch_train_dataset, prediction_model_valid_file_path)

  prediction_model_config_json = {
    "model_name_or_path": base_model if curr_iter == 0 else prediction_model,
    "tokenizer_name": base_model if curr_iter == 0 else prediction_model,
    "max_len": 512,
    "target_max_len": 64,
    "train_file_path": prediction_model_train_file_path,
    "valid_file_path": prediction_model_valid_file_path,
    "output_dir": prediction_model,
    "overwrite_output_dir": True,
    "per_device_train_batch_size": 10,
    "per_device_eval_batch_size": 10,
    "gradient_accumulation_steps": 6,
    "learning_rate": 1e-4,
    "num_train_epochs": epochs,
    "do_train": True,
    "do_eval": False,
    "prediction_loss_only": True,
    "remove_unused_columns": False,
    "save_strategy": 'no',
    "evaluation_strategy": 'no',
    "save_total_limit": 1,
    "load_best_model_at_end": True
  }

  # === START FINETUNE MODEL P ===
  finetune(prediction_model_config_json)


def convert_train_and_predict_data(esnli_path: str = 'e-SNLI/dataset/esnli_dev.csv'):
  ## convert data label to index
  # ['entailment', 'neutral', 'contradiction']
  dd = pd.read_csv(esnli_path)
  dd['gold_label_index'] = [0] * len(dd)
  dd['gold_label_index'][dd['gold_label'] == 'entailment'] = 0
  dd['gold_label_index'][dd['gold_label'] == 'neutral'] = 1
  dd['gold_label_index'][dd['gold_label'] == 'contradiction'] = 2
  dd['pair_index'] = dd.index
  len(dd[dd['gold_label_index'] == 0]), len(dd[dd['gold_label_index'] == 1]), len(
      dd[dd['gold_label_index'] == 2])

  dd = dd.sample(frac=1.0, random_state=42)
  train_df = dd.sample(frac=0.8, random_state=42)
  predict_df = dd[~dd['pair_index'].isin(train_df['pair_index'])]
  return train_df, predict_df


def random_al_iter(train_data, predict_data, column1='Sentence1',
                   column2='Sentence2', expl='Explanation_1',
                   label_column='gold_label_index', batch_num=50, epoch=2 , base_model='t5-base'):
  AL_result = []
  for i in range(3):
    select_for_single_train = train_data[:(i + 1) * batch_num]
    logger1.warning(f'##### {i}/{len(train_data) // batch_num + 1} ##### {len(select_for_single_train)}')
    one_iter(select_for_single_train,
             column_1=column1,
             column_2=column2,
             explanation_column=expl,
             label_column=label_column,
             rationale_model=rationale_model_path,
             prediction_model=prediction_model_path,
             curr_iter=i,
             epochs=epoch,
             base_model=base_model)

    r, p = [], []
    if i > (len(train_data) // batch_num) // 2:
      logger1.warning(f'##### predicting: {i} / {len(select_for_single_train)} ##### ')
      r, p = predict_pipeline(predict_data,
                              rationale_model_path=rationale_model_path,
                              predict_model_path=prediction_model_path,
                              column_1='Sentence1',
                              column_2='Sentence2',
                              explanation_column='Explanation_1',
                              label_column='gold_label_index')
      logger1.warning('Precision:{}'.format(precision_score(predict_data['gold_label'], p, average='weighted')))
      logger1.warning('Recall:{}'.format(recall_score(predict_data['gold_label'], p, average='weighted')))
      logger1.warning('Accuracy:{}'.format(accuracy_score(predict_data['gold_label'], p)))
      logger1.warning('F1 score:{}'.format(f1_score(predict_data['gold_label'], p, average='weighted')))
    AL_result.append({'select_data': select_for_single_train.to_dict('records'),
                      'predict_result': p,
                      'generated_explanation': r,
                      'iteration': i})
  return AL_result


def real_al_iter(train_data, predict_data, column1='Sentence1',
                 column2='Sentence2', expl='Explanation_1',
                 label_column='gold_label_index', batch_num=50,
                 epoch=2, iteration_num=2, base_model='t5-base'):

  iteration_num = num if (num := len(train_data) // batch_num < iteration_num) else iteration_num
  logger1.warning(f'total iteration num: {iteration_num}')
  AL_result = []
  remain_data = train_data
  previous_data = []
  data_for_train = []

  try:
    for i in range(iteration_num):
      selected_data, remain_data = similarity_batch_selection(train_data=remain_data,
                                                              previous_batch_train_data=previous_data,
                                                              criteria='even',
                                                              current_iter=i,
                                                              num_batch=batch_num,
                                                              column_1=column1,
                                                              column_2=column2,
                                                              explanation_column=expl)
      data_for_train = concatenate_datasets(dsets=[data_for_train, selected_data]) if data_for_train else selected_data
      logger1.warning(f'##### {i}/{iteration_num} ##### {len(data_for_train)}')
      one_iter(data_for_train,
               column_1=column1,
               column_2=column2,
               explanation_column=expl,
               label_column=label_column,
               rationale_model=rationale_model_path,
               prediction_model=prediction_model_path,
               curr_iter=i,
               epochs=epoch,
               base_model=base_model)

      previous_data = selected_data

      r, p = [], []

      logger1.warning(f'##### predicting: iteration-{i} , total data num {len(predict_data)} ##### ')
      r, p = predict_pipeline(predict_data,
                              rationale_model_path=rationale_model_path,
                              predict_model_path=prediction_model_path,
                              column_1='Sentence1',
                              column_2='Sentence2',
                              explanation_column='Explanation_1',
                              label_column='gold_label_index')
      AL_result.append({'select_data': data_for_train.to_dict(),
                        'predict_result': p,
                        'generated_explanation': r,
                        'iteration': i})

      logger1.warning('Precision:{}'.format(precision_score(predict_data['gold_label'], p, average='weighted')))
      logger1.warning('Recall:{}'.format(recall_score(predict_data['gold_label'], p, average='weighted')))
      logger1.warning('Accuracy:{}'.format(accuracy_score(predict_data['gold_label'], p)))
      logger1.warning('F1 score:{}'.format(f1_score(predict_data['gold_label'], p, average='weighted')))

  except Exception as e:
    logger1.error(e)
    print(e)

  finally:
    with open(f'train_result_{int(time.time())}.json', 'w') as f:
      json.dump(AL_result, f, indent=2)
  return AL_result


if __name__ == "__main__":

  # train_df, predict_df = convert_train_and_predict_data(esnli_path='data/esnli_dev.csv')
  # pd.read_json()
  # train_df.to_json('data/train.json', orient='split', index=False, indent=2)
  # predict_df.to_json('data/predict.json', orient='split', index=False, indent=2)

  # print('done')

  train_df = pd.read_json('data/train.json', orient='split')
  predict_df = pd.read_json('data/predict.json', orient='split')
  # random_al_iter(train_df, predict_df, base_model='google/t5-efficient-tiny')

  # r, p = predict_pipeline(predict_df[:50],
  #                         rationale_model_path,
  #                         prediction_model_path,
  #                         column_1='Sentence1',
  #                         column_2='Sentence2',
  #                         explanation_column='Explanation_1',
  #                         label_column='gold_label_index')
  # logger1.info('Precision:{}'.format(precision_score(predict_df[:50]['gold_label'], p, average='weighted')))
  # logger1.info('Recall:{}'.format(recall_score(predict_df[:50]['gold_label'], p, average='weighted')))
  # logger1.info('Accuracy:{}'.format(accuracy_score(predict_df[:50]['gold_label'], p)))
  # logger1.info('F1 score:{}'.format(f1_score(predict_df[:50]['gold_label'], p, average='weighted')))
  # print(r, p)

  real_al_iter(train_df, predict_df, batch_num=40, epoch=20, iteration_num=100, base_model='t5-base')

