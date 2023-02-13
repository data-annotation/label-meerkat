from services.model import tokenizer


def rationale_model_define_input(example, label: list):
  label_content = ""
  for i in range(len(label)):
    label_content += 'choice' + str(i + 1) + ': ' + str(label[i]) + ' '

  example['input_text'] = ('explain: what is the relationship between %s and %s ' % (example['hypothesis'], example['premise'])) + label_content
  example['target_text'] = '%s' % example['explanation_1']

  return example


def prediction_model_define_input(example, label: list):
  label_content = ""
  for i in range(len(label)):
    label_content += 'choice' + str(i + 1) + ': ' + str(label[i]) + ' '

  example['input_text'] = ('question: what is the relationship between %s and %s ' % (example['hypothesis'], example['premise'])) + label_content + (' <sep> because %s' % (example['generated_rationale']))
  example['target_text'] = '%s' % label[int(example['label'])]

  return example



def convert_to_features(example_batch):
    input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'], add_special_tokens=True, truncation=True, pad_to_max_length=True, max_length=512)
    target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], add_special_tokens=True, truncation=True, pad_to_max_length=True, max_length=64)

    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'target_ids': target_encodings['input_ids'],
        'target_attention_mask': target_encodings['attention_mask']
    }

    return encodings


def rationale_model_preprocessing(input_dataset):
  input_dataset = input_dataset.map(rationale_model_define_input, load_from_cache_file=False)

  # print(input_dataset[0])

  input_dataset = input_dataset.map(convert_to_features, batched=True, load_from_cache_file=False)

  columns = ['input_ids', 'target_ids', 'attention_mask', 'target_attention_mask']
  input_dataset.set_format(type='torch', columns=columns)
  return input_dataset


def prediction_model_preprocessing(input_dataset):
  input_dataset = input_dataset.map(prediction_model_define_input, load_from_cache_file=False)
  input_dataset = input_dataset.map(convert_to_features, batched=True, load_from_cache_file=False)

  columns = ['input_ids', 'target_ids', 'attention_mask', 'target_attention_mask']
  input_dataset.set_format(type='torch', columns=columns)
  return input_dataset