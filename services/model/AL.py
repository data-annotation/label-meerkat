import torch
from datasets import Dataset

from services.config import learning_rate
from services.config import num_epochs_p
from services.config import num_epochs_rg
from services.config import per_device_batch_size
from services.model.arch import finetune
from services.model.predictor import predict
from services.model.util.model_input import prediction_model_preprocessing
from services.model.util.model_input import rationale_model_preprocessing


def one_iter(labeled_data: [list, dict],
             last_model: str = None,
             model_id: str = None,
             curr_iter: int = 0):

  train_dataset = Dataset.from_dict(labeled_data) if isinstance(labeled_data, dict) else Dataset.from_list(labeled_data)
  rationale_model_batch_train_dataset = rationale_model_preprocessing(train_dataset,
                                                                      labels=['entailment', 'neutral', 'contradiction'],
                                                                      column_1='premise',
                                                                      column_2='hypothesis',
                                                                      explanation_column='explanation_1')
  torch.save(rationale_model_batch_train_dataset, "rationale_model_data/train_data.pt")
  torch.save(rationale_model_batch_train_dataset, "rationale_model_data/valid_data.pt")

  ## fine-tune rationale model
  # set config and load model (1st time load pretrained model)
  rationale_model_config_json = {}
  if curr_iter == 0:
    rationale_model_config_json = {
      "model_name_or_path": 't5-base',
      "tokenizer_name": 't5-base',
      "max_len": 512,
      "target_max_len": 64,
      "train_file_path": 'rationale_model_data/train_data.pt',
      "valid_file_path": 'rationale_model_data/valid_data.pt',
      "output_dir": 'rationale_model/',
      "overwrite_output_dir": True,
      "per_device_train_batch_size": per_device_batch_size,
      "per_device_eval_batch_size": per_device_batch_size,
      "gradient_accumulation_steps": 6,
      "learning_rate": learning_rate,
      "num_train_epochs": num_epochs_rg,
      "do_train": True,
      "do_eval": False,
      "prediction_loss_only": True,
      "remove_unused_columns": False,
      "save_strategy": 'no',
      "evaluation_strategy": 'no',
      "save_total_limit": 1,
      "load_best_model_at_end": True
    }
  else:
    rationale_model_config_json = {
      "model_name_or_path": 'rationale_model/',
      "tokenizer_name": 'rationale_model/',
      "max_len": 512,
      "target_max_len": 64,
      "train_file_path": 'rationale_model_data/train_data.pt',
      "valid_file_path": 'rationale_model_data/valid_data.pt',
      "output_dir": 'rationale_model/',
      "overwrite_output_dir": True,
      "per_device_train_batch_size": per_device_batch_size,
      "per_device_eval_batch_size": per_device_batch_size,
      "gradient_accumulation_steps": 6,
      "learning_rate": learning_rate,
      "num_train_epochs": num_epochs_rg,
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
                                'rationale_model/',
                                'ignore',
                                curr_iter)

  ## preprocess generated rationales
  prediction_model_batch_train_dataset = train_dataset.add_column("generated_rationale",
                                                                  predicted_rationale)
  prediction_model_batch_train_dataset = prediction_model_preprocessing(prediction_model_batch_train_dataset,
                                                                        labels=['entailment', 'neutral', 'contradiction'],
                                                                        column_1='premise',
                                                                        column_2='hypothesis',
                                                                        explanation_column='generated_rationale')
  torch.save(prediction_model_batch_train_dataset, 'prediction_model_data/train_data.pt')
  torch.save(prediction_model_batch_train_dataset, 'prediction_model_data/valid_data.pt')


  ## fine-tune prediction model
  # set config and load model (1st time load pretrained model)
  prediction_model_config_json = {}
  if curr_iter == 0:
    prediction_model_config_json = {
      "model_name_or_path": 't5-base',
      "tokenizer_name": 't5-base',
      "max_len": 512,
      "target_max_len": 64,
      "train_file_path": 'prediction_model_data/train_data.pt',
      "valid_file_path": 'prediction_model_data/valid_data.pt',
      "output_dir": 'prediction_model/',
      "overwrite_output_dir": True,
      "per_device_train_batch_size": per_device_batch_size,
      "per_device_eval_batch_size": per_device_batch_size,
      "gradient_accumulation_steps": 6,
      "learning_rate": 1e-4,
      "num_train_epochs": num_epochs_p,
      "do_train": True,
      "do_eval": False,
      "prediction_loss_only": True,
      "remove_unused_columns": False,
      "save_strategy": 'no',
      "evaluation_strategy": 'no',
      "save_total_limit": 1,
      "load_best_model_at_end": True
    }
  else:
    prediction_model_config_json = {
      "model_name_or_path": 'prediction_model/',
      "tokenizer_name": 'prediction_model/',
      "max_len": 512,
      "target_max_len": 64,
      "train_file_path": 'prediction_model_data/train_data.pt',
      "valid_file_path": 'prediction_model_data/valid_data.pt',
      "output_dir": 'prediction_model/',
      "overwrite_output_dir": True,
      "per_device_train_batch_size": per_device_batch_size,
      "per_device_eval_batch_size": per_device_batch_size,
      "gradient_accumulation_steps": 6,
      "learning_rate": 1e-4,
      "num_train_epochs": num_epochs_p,
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



