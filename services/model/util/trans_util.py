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


