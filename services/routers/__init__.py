from sentence_transformers import SentenceTransformer
from services.orm.tables import engine


class EncodeModel:
  _model = None

  @property
  def model(self):
    if not self._model:
      print('init encode model')
      self._model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return self._model


encode_model = EncodeModel()
