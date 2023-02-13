import torch
from transformers import T5Tokenizer
from sentence_transformers import SentenceTransformer, util

from services.config import SentenceTransformer_PATH


device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = T5Tokenizer.from_pretrained('t5-base')
tokenizer.add_special_tokens({'sep_token': '<sep>'})

model_SentenceTransformer = SentenceTransformer(SentenceTransformer_PATH)



