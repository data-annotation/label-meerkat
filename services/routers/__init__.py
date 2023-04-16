from typing import Dict
from typing import List

from sentence_transformers import SentenceTransformer


class EncodeModel:
  _model = None

  @property
  def model(self):
    if not self._model:
      print('init encode model')
      self._model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return self._model


encode_model = EncodeModel()


def select_model_for_train(models: List[Dict]):
  model_num = len(models)
  model_for_train = None

  for m in models:
      if m['status'] == 0:
          model_for_train = m
          break

  return model_for_train, model_num



