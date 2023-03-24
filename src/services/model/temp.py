from rouge import Rouge
rouge = Rouge()

SentenceTransformer_PATH = "/content/all-MiniLM-L6-v2"


import torch
from transformers import T5Tokenizer
from sentence_transformers import SentenceTransformer, util

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Device: ", device)

tokenizer = T5Tokenizer.from_pretrained('t5-base')
tokenizer.add_special_tokens({'sep_token': '<sep>'})

model_SentenceTransformer = SentenceTransformer(SentenceTransformer_PATH)
