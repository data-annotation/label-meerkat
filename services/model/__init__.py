import torch
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer

from services.config import SentenceTransformer_PATH


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Tokenizer:
  worker = None

  @property
  def t(self):
    if not self.worker:
      self.worker = T5Tokenizer.from_pretrained('t5-base')
      self.worker.add_special_tokens({'sep_token': '<sep>'})
    return self.worker


tokenizer = Tokenizer()
# tokenizer = T5Tokenizer.from_pretrained('t5-base')
# tokenizer.add_special_tokens({'sep_token': '<sep>'})

model_SentenceTransformer = SentenceTransformer(SentenceTransformer_PATH,
                                                cache_folder=SentenceTransformer_PATH)



