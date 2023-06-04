import pandas as pd
from datasets import Dataset


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


def trans_data_to_dataframe(raw_data) -> pd.DataFrame:
  if isinstance(raw_data, pd.DataFrame):
    return raw_data
  elif isinstance(raw_data, dict):
    trans_dataset = pd.DataFrame(**raw_data)
  elif isinstance(raw_data, list):
    trans_dataset = pd.DataFrame(raw_data)
  else:
    trans_dataset = raw_data
  return trans_dataset


def label2id(labels: list, label_list: list = None):
  label_list = label_list or sorted(set(labels))
  id_mapping = {l: i for i, l in enumerate(label_list)}
  return [id_mapping[label] for label in labels]


def id2label(labels: list, label_list: list):
  id_mapping = {i: l for i, l in enumerate(label_list)}
  return [id_mapping.get(label, label) for label in labels]
