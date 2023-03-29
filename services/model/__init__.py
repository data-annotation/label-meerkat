import torch
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer

from services.config import SentenceTransformer_PATH


device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = T5Tokenizer.from_pretrained('t5-base')
tokenizer.add_special_tokens({'sep_token': '<sep>'})

model_SentenceTransformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',
                                                cache_folder=SentenceTransformer_PATH)



