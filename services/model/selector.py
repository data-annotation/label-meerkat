import random
from typing import Union

import datasets
import pandas as pd
from datasets import Dataset

import torch
from sentence_transformers import util

from services.const import UnlabeledDataSelectWay
from services.model import device
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


def similarity_batch_selection_v2(for_select_data: Union[list, pd.DataFrame, Dataset, dict],
                                  similarity_by: list = None,
                                  labeled_data: Union[list, pd.DataFrame, Dataset, dict] = None,
                                  labeled_column: list = None,
                                  num_batch: int = 20,
                                  select_way: str = UnlabeledDataSelectWay.combined) -> (datasets.Dataset, datasets.Dataset):
  """
  Args:
    for_select_data: 要选择的数据
    similarity_by: 相似度比较字段
    labeled_data: 已标注的数据，不传时则仅在 for_select_data 中进行相似的比较
    labeled_column: 标签字段
    num_batch: 数据选择数量
    select_way: 数据选择方式，combined，从排序后的索引列表中选择得分最高的一半和得分最低的一半作为选中的样本索引
                           even，均匀地从排序后的数据中选择样本索引

  Returns:
    从for_select_data选择出的Dataset, for_select_data中剩余的Dataset
  """
  similarity_by = similarity_by or ['premise', 'hypothesis']
  labeled_column = labeled_column or similarity_by
  for_select_data = trans_data_to_dataset(for_select_data)
  labeled_data = trans_data_to_dataset(labeled_data)

  for_select_data_embed = model_SentenceTransformer.encode([' '.join([i[k] for k in similarity_by]) for i in for_select_data],
                                                           convert_to_tensor=True).to(device)
  labeled_data_embed = None
  if labeled_data:
    labeled_data_embed = model_SentenceTransformer.encode([' '.join([i[k] for k in labeled_column]) for i in labeled_data],
                                                          convert_to_tensor=True).to(device)
  labeled_data_embed = labeled_data_embed if labeled_data_embed is not None else for_select_data_embed
  cosine_scores = util.cos_sim(for_select_data_embed, labeled_data_embed)
  mean_cosine_scores = torch.mean(cosine_scores, 1, True)
  mean_cosine_scores_list = [i.item() for i in mean_cosine_scores]
  mean_cosine_scores_list_sorted_with_idx = sorted(range(len(mean_cosine_scores_list)),
                                                   key=lambda i: mean_cosine_scores_list[i])

  if select_way == 'combined':
    sampled_idx = mean_cosine_scores_list_sorted_with_idx[-int(num_batch / 2):] + mean_cosine_scores_list_sorted_with_idx[:int(num_batch / 2)]
  else:
    step = int(len(for_select_data) / num_batch)
    sampled_idx = mean_cosine_scores_list_sorted_with_idx[0::step]

  remaining_idx = list(range(len(for_select_data)))
  for i in sampled_idx:
    remaining_idx.remove(i)

  return for_select_data.select(sampled_idx), for_select_data.select(remaining_idx)


if __name__ == "__main__":

  for_select = pd.read_csv('../../data/esnli_for_dev.csv')
  label_data = pd.DataFrame({
    "label": [1, 2, 0, 1, 0, 2, 2, 0, 1, 1],
    "id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "explanation": ["the person is not necessarily training his horse",
                    "One cannot be on a jumping horse cannot be a diner ordering food.",
                    "a broken down airplane is outdoors",
                    "Just because they are smiling and waving at a camera does not imply their parents or anyone is anyone behind it",
                    "The children must be present to see them smiling and waving.",
                    "One cannot be smiling and frowning at the same time.",
                    "One cannot be in the middle of a bridge if they are on the sidewalk.",
                    "jumping on skateboard is the same as doing trick on skateboard.",
                    "Just because the boy is jumping on a skateboard does not imply he is wearing safety equipment",
                    "it is not necessarily true the man drinks his juice"]
  })
  project_with_label = for_select.merge(label_data, how='left', on='id')
  label_data = project_with_label[project_with_label['label'].notnull()]
  for_select = project_with_label[project_with_label['label'].isnull()]
  selected_data, remain_data = similarity_batch_selection_v2(for_select_data=for_select,
                                                             similarity_by=['sentence1', 'sentence2'])
  print(selected_data, remain_data)
