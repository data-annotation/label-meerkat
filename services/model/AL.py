import os.path
from typing import Union

import pandas as pd
import torch
from datasets import Dataset

from services.config import base_model
from services.config import learning_rate
from services.config import model_path
from services.config import num_epochs_p
from services.config import num_epochs_rg
from services.config import per_device_batch_size
from services.model.arch import finetune
from services.model.predictor import predict
from services.model.util.model_input import prediction_model_preprocessing
from services.model.util.model_input import rationale_model_preprocessing


def one_training_iteration(labeled_data: Union[list, dict],
                           column_1: str = 'premise',
                           column_2: str = 'hypothesis',
                           explanation_column: str = 'explanation_1',
                           labels: list = None,
                           model_id: str = 'test_model',
                           old_model_id: str = None):
    labels = labels or ['entailment', 'neutral', 'contradiction']

    if isinstance(labeled_data, pd.DataFrame):
        train_dataset = Dataset.from_pandas(labeled_data)
    elif isinstance(labeled_data, dict):
        train_dataset = Dataset.from_dict(labeled_data)
    elif isinstance(labeled_data, list):
        train_dataset = Dataset.from_list(labeled_data)
    else:
        train_dataset = labeled_data

    work_dir = os.path.join(model_path, model_id)
    last_model = os.path.join(model_path, old_model_id or '/', 'model')
    data_path = os.path.join(work_dir, 'data')
    r_model_path = os.path.join(work_dir, 'model', 'rationale')
    p_model_path = os.path.join(work_dir, 'model', 'prediction')

    os.makedirs(r_model_path, exist_ok=True)
    os.makedirs(p_model_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)

    rationale_model_batch_train_dataset = rationale_model_preprocessing(train_dataset,
                                                                        labels=labels,
                                                                        column_1=column_1,
                                                                        column_2=column_2,
                                                                        explanation_column=explanation_column)

    rationale_model_train_file_path = os.path.join(data_path, 'rationale_train_data.pt')
    rationale_model_valid_file_path = os.path.join(data_path, 'rationale_valid_data.pt')
    torch.save(rationale_model_batch_train_dataset, rationale_model_train_file_path)
    torch.save(rationale_model_batch_train_dataset, rationale_model_valid_file_path)

    # fine-tune rationale model
    # set config and load model (1st time load pretrained model)

    rationale_model_config_json = {
        "model_name_or_path": os.path.join(last_model, 'rationale') if old_model_id else base_model,
        "tokenizer_name": os.path.join(last_model, 'rationale') if old_model_id else base_model,
        "max_len": 512,
        "target_max_len": 64,
        "train_file_path": rationale_model_train_file_path,
        "valid_file_path": rationale_model_valid_file_path,
        "output_dir": r_model_path,
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

    # print("=== START FINETUNE MODEL RG ===")

    finetune(rationale_model_config_json, model_id=model_id, total_steps=2, current_step=1)

    # print("=== START GENERATE RATIONALE FOR MODEL P ===")

    predicted_rationale = predict(rationale_model_batch_train_dataset,
                                  'rationale',
                                  r_model_path)

    # preprocess generated rationales
    prediction_model_batch_train_dataset = train_dataset.add_column("generated_rationale",
                                                                    predicted_rationale)
    print(predicted_rationale)
    prediction_model_batch_train_dataset = prediction_model_preprocessing(prediction_model_batch_train_dataset,
                                                                          labels=labels,
                                                                          column_1=column_1,
                                                                          column_2=column_2,
                                                                          explanation_column=explanation_column)
    prediction_model_train_file_path = os.path.join(data_path, 'prediction_train_data.pt')
    prediction_model_valid_file_path = os.path.join(data_path, 'prediction_valid_data.pt')
    torch.save(prediction_model_batch_train_dataset, prediction_model_train_file_path)
    torch.save(prediction_model_batch_train_dataset, prediction_model_valid_file_path)

    prediction_model_config_json = {
        "model_name_or_path": os.path.join(last_model, 'prediction') if old_model_id else base_model,
        "tokenizer_name": os.path.join(last_model, 'prediction') if old_model_id else base_model,
        "max_len": 512,
        "target_max_len": 64,
        "train_file_path": prediction_model_train_file_path,
        "valid_file_path": prediction_model_valid_file_path,
        "output_dir": p_model_path,
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
    finetune(prediction_model_config_json, model_id=model_id, total_steps=2, current_step=2)
