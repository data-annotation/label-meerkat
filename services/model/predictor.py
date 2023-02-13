import torch
from tqdm.auto import tqdm
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer

from services.config import device


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
    print("RESULTS: [# current iteration %d], [# data per label %d], [# epoch RG %d], [# epoch P %d], [Learning Rate %f], [Batch size per device %d], [Select %s], [Accuracy %f]" % (iteration, num_data_per_batch, num_epochs_rg, num_epochs_p, learning_rate, per_device_batch_size, flag, accuracy))


# def rationale_model_accuracy(predictions, references):
#   score = total = 0
#   for groundtruths, prediction in zip(references, predictions):
#     total += 1

#     res = scorer.score(groundtruths, prediction)['rougeL'].fmeasure

#     score += res

#   print("rationale model ROUGE-L: ", score/len(predictions) )


def predict(test_dataset, model_name, model_path, flag, iteration):
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
    prediction_model_accuracy(predictions, references, flag, iteration)
  ##elif model_name == 'rationale':
  ##  rationale_model_accuracy(predictions, references)


  return predictions

