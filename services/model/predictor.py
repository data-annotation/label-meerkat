from typing import Union

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer

from services.model import device
import meerkat as mk


def prediction_model_accuracy(predictions, references, flag, iteration):
  correct = total = 0

  for groundtruths, prediction in zip(references, predictions):
    total += 1

    if groundtruths == prediction:
      correct += 1
  accuracy = correct / total
  print("total test examples: ", total)
  print("prediction model accuracy: ", accuracy)
  if flag != 'ignore':
    print("RESULTS: [# current iteration %d], [# data per label %d], [# epoch RG %d], [# epoch P %d], [Learning Rate %f], "
          "[Batch size per device %d], [Select %s], [Accuracy %f]" % (iteration, num_data_per_batch,
                                                                      num_epochs_rg, num_epochs_p,
                                                                      learning_rate, per_device_batch_size,
                                                                      flag, accuracy))


def prediction_accuracy(predictions, references):
  correct = total = 0

  for groundtruths, prediction in zip(references, predictions):
    total += 1

    if groundtruths == prediction:
      correct += 1
  accuracy = correct / total
  print("total test examples: ", total)
  print("prediction model accuracy: ", accuracy)


def predict(test_dataset, model_name, model_path):
  model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
  tokenizer = T5Tokenizer.from_pretrained(model_path)

  # os.makedirs(TEMP_FOLDER_PATH + 'rationale_model_' + str(index) )
  # os.makedirs(TEMP_FOLDER_PATH + 'prediction_model_' + str(index) )
  # shutil.rmtree(TEMP_FOLDER_PATH+'rationale_model_'+str(index))
  # shutil.rmtree(TEMP_FOLDER_PATH+'prediction_model_'+str(index))
  '''
  ## CLEANING CODE

  shutil.rmtree(model_path)

  if not os.path.exists(model_path):
    os.makedirs(model_path)

  model.save_pretrained(model_path) 
  ##tokenizer.save_pretrained(model_path)
  print(" - successfully clean model saving folder")
  '''
  dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

  answers = []
  for batch in tqdm(dataloader):
    outs = model.generate(input_ids=batch['input_ids'].to(device),
                          attention_mask=batch['attention_mask'].to(device),
                          max_length=64,
                          early_stopping=True)
    outs = [tokenizer.decode(ids, skip_special_tokens=True).encode('utf-8').decode('utf-8') for ids in outs]
    answers.extend(outs)

  predictions = []
  references = []
  for ref, pred in zip(test_dataset, answers):
    predictions.append(pred)
    # references.append(ref['answer'])

    references.append(tokenizer.decode(ref['target_ids'], skip_special_tokens=True))

  print("1st predicted:", predictions[0])
  print("1st groundtruth:", references[0])
  assert len(predictions) == len(references)
  # print(len(predictions), len(references))

  if model_name == 'prediction':
    prediction_accuracy(predictions, references)
  ##elif model_name == 'rationale':
  ##  rationale_model_accuracy(predictions, references)

  return predictions


def save_predictions(predictions: Union[list, pd.DataFrame],
                     label_column: str = 'label',
                     save_path: str = None,
                     merge_by:str = None,
                     origin_data: Union[list, pd.DataFrame] = None):
  """
  save predict result
  Args:
    predictions: predict result, list of dict like Dataframe.to_dict('records') or Dataframe
    label_column
    save_path: path to save
    merge_by: the column used for merge prediction and origin data
    origin_data: list of dict like Dataframe.to_dict('records') or Dataframe

  Returns:

  """
  # type cast
  if isinstance(predictions, list):
    if isinstance(predictions[0], dict):
      predictions = pd.DataFrame(predictions)
    else:
      predictions = pd.DataFrame(predictions, columns=[label_column])

  if origin_data:
    origin_data = pd.DataFrame(origin_data) if isinstance(origin_data, list) else origin_data
    if merge_by:
      merged_data = origin_data.merge(predictions, how='left', on=label_column)
    else:
      merged_data = origin_data.assign(**{label_column: predictions.iloc[:, 0]})
  else:
    merged_data = predictions
  mk.from_pandas(merged_data, index=False).write(save_path)
