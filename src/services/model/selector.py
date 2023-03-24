import random

from .util.dataset_concatenate import custom_concatenate
from .util.dataset_concatenate import custom_concatenate_2


def random_batch_selection(train_dataset, num_batch):
  sampled_idx = random.sample( list( range(len(train_dataset)) ), num_batch )

  remaining_idx = list( range(len(train_dataset)) )
  for i in sampled_idx:
    remaining_idx.remove(i)

  # batch_dataset = train_dataset.select(sampled_idx)
  # remain_dataset = train_dataset.select(remaining_idx)
  #return batch_dataset, remain_dataset
  return train_dataset.select(sampled_idx), train_dataset.select(remaining_idx)


def batch_selection(train_dataset, num_batch, criteria, current_iter, previous_batch_train_dataset):
  ## we are pre-splitting data by label
  ## which is not exactly what the real world should be
  train_dataset_zero = train_dataset.filter(lambda example: example["label"] == 0)
  train_dataset_one = train_dataset.filter(lambda example: example["label"] == 1)
  train_dataset_two = train_dataset.filter(lambda example: example["label"] == 2)
  print(len(train_dataset_zero), len(train_dataset_one), len(train_dataset_two))

  if criteria == 'random':
    batch_train_dataset_zero, remain_train_dataset_zero = random_batch_selection(train_dataset_zero,
                                                                                 num_batch)
    batch_train_dataset_one, remain_train_dataset_one = random_batch_selection(train_dataset_one, num_batch)
    batch_train_dataset_two, remain_train_dataset_two = random_batch_selection(train_dataset_two, num_batch)
  # else:
  #   batch_train_dataset_zero, remain_train_dataset_zero = similarity_batch_selection(train_dataset_zero, num_batch, criteria, current_iter, previous_batch_train_dataset)
  #   batch_train_dataset_one, remain_train_dataset_one = similarity_batch_selection(train_dataset_one, num_batch, criteria, current_iter, previous_batch_train_dataset)
  #   batch_train_dataset_two, remain_train_dataset_two = similarity_batch_selection(train_dataset_two, num_batch, criteria, current_iter, previous_batch_train_dataset)


  batch_dataset = custom_concatenate(batch_train_dataset_zero,
                                     batch_train_dataset_one,
                                     batch_train_dataset_two)
  remain_dataset = custom_concatenate(remain_train_dataset_zero,
                                      remain_train_dataset_one,
                                      remain_train_dataset_two)
  # previous_batch_train_dataset = copy.deepcopy(batch_dataset)
  # previous_batch_train_dataset = batch_dataset
  previous_batch_train_dataset = custom_concatenate_2(batch_dataset, previous_batch_train_dataset)

  ##batch_dataset = batch_dataset.remove_columns(['Unnamed: 0'])

  print(batch_dataset)
  print('---')
  print(remain_dataset)

  print("Finish batch selection by ", criteria, len(batch_dataset), len(remain_dataset))

  return batch_dataset, remain_dataset, previous_batch_train_dataset
