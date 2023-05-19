# import logging
# from tqdm import tqdm
# import numpy as np
# from numpy import ndarray
# import torch
# from torch import Tensor, device
# import transformers
# from transformers import GPT2Tokenizer, GPT2Model
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import normalize
# from typing import List, Dict, Tuple, Type, Union

# # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# # model = GPT2Model.from_pretrained('gpt2')
# # model =  model.to("cuda")
# # def encode(sentence: Union[str, List[str]]) -> Union[ndarray, Tensor]:
    
# #     with torch.no_grad():
        
# #         inputs = tokenizer.encode(
# #             sentence,
# #             return_tensors = "pt"
# #         ).to("cuda")
        
# #         outputs = model(inputs)
# #         embeddings = outputs[0].cpu()
            
# #     return embeddings.numpy()
    


# # def similarity(queries: Union[str, List[str]],
# #                 keys: Union[str, List[str], ndarray],
# #                 ) -> Union[float, ndarray]:
# #     query_vecs = encode(queries)
# #     key_vecs = encode(keys)
    
    
# #     query_vecs = query_vecs.reshape(1, -1)
# #     key_vecs = key_vecs.reshape(1, -1)
    
# #     similarities = cosine_similarity(query_vecs, key_vecs)
    
# #     if single_query:
# #         similarities = similarities[0]
# #         if single_key:
# #             similarities = float(similarities[0])
# #     return similarities

# # sentences_a = 'A woman is reading.'
# # sentences_b = 'A woman is making a photo.'
# # similarities = similarity(sentences_a, sentences_b)
# # print(similarities)

# # # from transformers import GPT2Tokenizer, GPT2Model

# # # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# # # model = GPT2Model.from_pretrained('gpt2')

# # # text = "This is a sentence."
# # # input_ids = tokenizer.encode(text, return_tensors='pt')
# # # outputs = model(input_ids)
# # # last_hidden_states = outputs[0]

# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2Model.from_pretrained('gpt2')
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# texts = ["This is a sentence.", "This is another sentence."]
# with torch.no_grad():
#     input_ids = tokenizer.encode(texts, return_tensors='pt', padding=True)
#     outputs = model(input_ids)
#     last_hidden_states = outputs[0]
# print(last_hidden_states)
# print(last_hidden_states.shape)

# query_vecs = last_hidden_states[0][0].numpy()
# key_vecs = last_hidden_states[0][1].numpy()


# query_vecs = query_vecs.reshape(1, -1)
# key_vecs = key_vecs.reshape(1, -1)

# similarities = cosine_similarity(query_vecs, key_vecs)


# similarities = float(similarities[0])

# print(similarities)


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

    model_name = 'gpt2'
    
    model_save_path = train(sts_dataset_path)
    predict(model_save_path, test_samples)