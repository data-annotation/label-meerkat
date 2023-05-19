# import logging
# from tqdm import tqdm
# import numpy as np
# from numpy import ndarray
# import torch
# from torch import Tensor, device
# import transformers
# from transformers import AutoModel, AutoTokenizer
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import normalize
# from typing import List, Dict, Tuple, Type, Union

# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# model = AutoModel.from_pretrained('bert-base-uncased')
# model =  model.to("cuda")
# def encode(sentence: Union[str, List[str]],
#             return_numpy: bool = False,
#             normalize_to_unit: bool = True,
#             keepdim: bool = False,
#             batch_size: int = 64,
#             max_length: int = 128) -> Union[ndarray, Tensor]:
    
#     single_sentence = False

#     if isinstance(sentence, str):
#         sentence = [sentence]
#         single_sentence = True
    
#     embedding_list = []
#     with torch.no_grad():
#         total_batch = len(sentence) // batch_size + (1 if len(sentence) % batch_size > 0 else 0)
#         for batch_id in tqdm(range(total_batch)):
#             inputs = tokenizer(
#                 sentence[batch_id*batch_size:(batch_id+1)*batch_size],
#                 padding = True,
#                 truncation = True,
#                 max_length = max_length,
#                 return_tensors = "pt"
#             )
#             inputs = {k:v.to("cuda") for k,v in inputs.items()}
#             outputs = model(**inputs, return_dict = True)
#             embeddings = outputs.pooler_output
#             if normalize_to_unit:
#                 embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
#             embedding_list.append(embeddings.cpu())
#     embeddings = torch.cat(embedding_list, 0)

#     if single_sentence and not keepdim:
#         embeddings = embeddings[0]
    
#     if return_numpy and not isinstance(embeddings, ndarray):
#         return embeddings.numpy()
#     return embeddings
    


# def similarity(queries: Union[str, List[str]],
#                 keys: Union[str, List[str], ndarray],
#                 ) -> Union[float, ndarray]:
#     query_vecs = encode(queries, return_numpy=True)
#     if not instance(keys, ndarray):
#         key_vecs = encode(keys, return_numpy=True)
#     else:
#         key_vecs = keys
    
#     single_query, single_key = len(query_vecs.shape) == 1, len(key_vecs.shape) == 1
#     if single_query:
#         query_vecs = query_vecs.reshape(1, -1)
#     if single_key:
#         key_vecs = key_vecs.reshape(1, -1)
#     similarities = cosine_similarity(query_vecs, key_vecs)
#     if single_query:
#         similarities = similarities[0]
#         if single_key:
#             similarities = float(similarities[0])
#     return similarities

# sentences_a = ['A woman is reading.']
# sentences_b = ['A woman is making a photo.']
# similarities = similarity(sentences_a, sentences_b)
# print(similarities)


"""
This examples trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) for the STSbenchmark from scratch. It generates sentence embeddings
that can be compared using cosine-similarity to measure the similarity.

Usage:
python training_nli.py

OR
python training_nli.py pretrained_transformer_model_name
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import sys
import os
import gzip
import csv

def train(sts_dataset_path, batch_size = 16, num_epochs = 4):
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)

    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    word_embedding_model = models.Transformer(model_name)
    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    train_loss = losses.CosineSimilarityLoss(model=model)

    model_save_path = 'model.pth'

    logging.info("Read STSbenchmark dev dataset")
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')


    # Configure the training. We skip evaluation in this example
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs  * 0.1) #10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))


    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=1000,
            warmup_steps=warmup_steps,
            output_path=model_save_path)

    return model_save_path
##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################
def predict(model_save_path, test_samples):
    model = SentenceTransformer(model_save_path)
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
    test_evaluator(model, output_path=model_save_path)

if __name__ == '__main__':
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout



    #Check if dataset exsist. If not, download and extract  it
    sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'

    if not os.path.exists(sts_dataset_path):
        util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

    # Convert the dataset to a DataLoader ready for training
    logging.info("Read STSbenchmark train dataset")

    train_samples = []
    dev_samples = []
    test_samples = []
    with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
            inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)

            if row['split'] == 'dev':
                dev_samples.append(inp_example)
            elif row['split'] == 'test':
                test_samples.append(inp_example)
            else:
                train_samples.append(inp_example)

    model_name = 'bert-base-uncased'
    
    model_save_path = train(sts_dataset_path)
    predict(model_save_path, test_samples)