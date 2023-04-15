from datasets import Dataset


def custom_concatenate(dataset1, dataset2, dataset3):
  new_dataset = {}

  for key in dataset1.features.keys():
    if key != 'Unnamed: 0':
      new_dataset[key] = []

  for example in dataset1:
    for key in example:
      if key != 'Unnamed: 0':
        new_dataset[key].append(example[key])

  for example in dataset2:
    for key in example:
      if key != 'Unnamed: 0':
        new_dataset[key].append(example[key])

  for example in dataset3:
    for key in example:
      if key != 'Unnamed: 0':
        new_dataset[key].append(example[key])

  new_dataset = Dataset.from_dict(new_dataset)

  return new_dataset


def custom_concatenate_2(dataset1, dataset2):
  new_dataset = {}

  for key in dataset1.features.keys():
    if key != 'Unnamed: 0':
      new_dataset[key] = []

  for example in dataset1:
    for key in example:
      if key != 'Unnamed: 0':
        new_dataset[key].append(example[key])

  for example in dataset2:
    for key in example:
      if key != 'Unnamed: 0':
        new_dataset[key].append(example[key])

  new_dataset = Dataset.from_dict(new_dataset)

  return new_dataset