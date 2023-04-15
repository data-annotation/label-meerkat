import random

import torch
from sentence_transformers import util

from services.model import device
from services.model.util.dataset_concatenate import custom_concatenate
from services.model.util.dataset_concatenate import custom_concatenate_2
from services.model.util.trans_util import trans_data_to_dataset
from services.model import model_SentenceTransformer


def random_batch_selection(train_dataset, num_batch):
  sampled_idx = random.sample( list( range(len(train_dataset)) ), num_batch )

  remaining_idx = list( range(len(train_dataset)) )
  for i in sampled_idx:
    remaining_idx.remove(i)

  # batch_dataset = train_dataset.select(sampled_idx)
  # remain_dataset = train_dataset.select(remaining_idx)
  #return batch_dataset, remain_dataset
  return train_dataset.select(sampled_idx), train_dataset.select(remaining_idx)


def similarity_batch_selection(train_data,
                               previous_batch_train_data,
                               criteria: str,
                               current_iter: int = 0,
                               num_batch: int = 20,
                               column_1: str = 'premise',
                               column_2: str = 'hypothesis',
                               explanation_column: str = 'explanation_1'):
  # logger1.warning(f'start select data by similarity, to_select: {len(train_data)}, prev_selected: {len(previous_batch_train_data)}')
  train_dataset = trans_data_to_dataset(train_data)
  previous_batch_train_dataset = trans_data_to_dataset(previous_batch_train_data)

  ## THESE TWO METHODS ONLY COMPARE THE TASK CONTENT
  if criteria in ['combined', 'even']:

    ## During the first iteration, there's no human annotattion
    ## So we compare the similarity between original content
    if current_iter == 0:
      sentence_list = [' '.join([i[column_2], i[column_1]]) for i in train_dataset]
      rationale_list = [' '.join([i[column_2], i[column_1]]) for i in train_dataset]
    else:
      sentence_list = [' '.join([i[column_2], i[column_1]]) for i in train_dataset]
      rationale_list = [' '.join([i[column_2], i[column_1]]) for i in previous_batch_train_dataset]


    sentence_list_embeddings = model_SentenceTransformer.encode(sentence_list, convert_to_tensor=True).to(
      device)
    rationale_list_embeddings = model_SentenceTransformer.encode(rationale_list, convert_to_tensor=True).to(
      device)

    cosine_scores = util.cos_sim(sentence_list_embeddings, rationale_list_embeddings)

    mean_cosine_scores = torch.mean(cosine_scores, 1, True)
    mean_cosine_scores_list = [i.item() for i in mean_cosine_scores]
    mean_cosine_scores_list_sorted_with_idx = sorted(range(len(mean_cosine_scores_list)),
                                                     key=lambda i: mean_cosine_scores_list[i])

    # combine half top ranked ones and half bottom ranked ones
    if criteria == 'combined':
      sampled_idx = mean_cosine_scores_list_sorted_with_idx[
                    -int(num_batch / 2):] + mean_cosine_scores_list_sorted_with_idx[:int(num_batch / 2)]
    # evenly select examples from ranked data
    elif criteria == 'even':
      step = int(len(train_dataset) / num_batch)
      sampled_idx = mean_cosine_scores_list_sorted_with_idx[0::step]

  ## THESE TWO METHODS COMPARE THE RATIONALES AND TASK CONTENT
  elif criteria in ['combined_rationale', 'even_rationale']:

    ## During the first iteration, there's no human annotattion
    ## So we compare the similarity between original content
    if current_iter == 0:
      sentence_list = [' '.join([i[column_2], i[column_1]]) for i in train_dataset]
      rationale_list = [' '.join([i[column_2], i[column_1]]) for i in train_dataset]
    else:
      sentence_list = [' '.join([i[column_2], i[column_1]]) for i in train_dataset]
      rationale_list = [i[explanation_column] for i in previous_batch_train_dataset]

    sentence_list_embeddings = model_SentenceTransformer.encode(sentence_list, convert_to_tensor=True).to(
      device)
    rationale_list_embeddings = model_SentenceTransformer.encode(rationale_list, convert_to_tensor=True).to(
      device)

    cosine_scores = util.cos_sim(sentence_list_embeddings, rationale_list_embeddings)

    mean_cosine_scores = torch.mean(cosine_scores, 1, True)
    mean_cosine_scores_list = [i.item() for i in mean_cosine_scores]
    mean_cosine_scores_list_sorted_with_idx = sorted(range(len(mean_cosine_scores_list)),
                                                     key=lambda i: mean_cosine_scores_list[i])

    if criteria == 'combined_rationale':
      sampled_idx = mean_cosine_scores_list_sorted_with_idx[
                    -int(num_batch / 2):] + mean_cosine_scores_list_sorted_with_idx[:int(num_batch / 2)]
    elif criteria == 'even_rationale':
      step = int(len(train_dataset) / num_batch)
      sampled_idx = mean_cosine_scores_list_sorted_with_idx[0::step]

  ## finish sampled index selection
  remaining_idx = list(range(len(train_dataset)))
  for i in sampled_idx:
    remaining_idx.remove(i)

  # logger1.warning(f'End select data by similarity, selected: {len(sampled_idx)}, remain: {len(remaining_idx)}')
  return train_dataset.select(sampled_idx), train_dataset.select(remaining_idx)

