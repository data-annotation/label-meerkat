import pandas as pd
import torch
from transformers import BartTokenizer, BartForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from services.model.util.trans_util import label2id, id2label
from services.model import device


def load_from_pretrained(model_path: str = 'facebook/bart-base',
                         num_labels: int = 2):
    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = BartForSequenceClassification.from_pretrained(model_path,
                                                          num_labels=num_labels)
    return model, tokenizer


def train(data: [list, pd.DataFrame],
          labels,
          batch_size=10,
          num_epochs=5,
          old_model='facebook/bart-base',
          output_model='test_model',
          label_list=None,
          device=device):
    labels = label2id(labels, label_list=label_list)
    model, tokenizer = load_from_pretrained(old_model, len(label_list))
    train_dataset = Dataset.from_dict({'text': data, 'label': labels})
    train_dataset = train_dataset.map(lambda example: tokenizer(example['text'],
                                                                padding='max_length',
                                                                truncation=True,
                                                                max_length=512),
                                      batched=True)
    train_dataset.set_format('torch',
                             columns=['input_ids', 'attention_mask', 'label'],
                             device=device)

    training_args = TrainingArguments(
        output_dir=output_model,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        logging_dir='./logs',
        logging_steps=10,
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset
    )

    trainer.train()
    tokenizer.save_pretrained(output_model)
    model.save_pretrained(output_model)
    return output_model


def predict(data: [list, pd.DataFrame],
            model_path: str = 'test_model',
            num_label: int = 2,
            device: str = device):
    model, tokenizer = load_from_pretrained(model_path, num_label)
    model.to(device)
    model.eval()

    predictions = []
    for text in data:
        encoding = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions.append(torch.argmax(logits).item())

    return predictions


if __name__ == "__main__":
    data = ['This is a positive sentence.', 'I\'m feeling great today.', 'Negative feedback received.',
            'The product is not good.']
    labels = ['positive', 'positive', 'negative', 'positive']

    model_path = train(data, labels, num_epochs=1, label_list=['positive', 'negative'])

    predictions = predict(['I feel good', "it's so bad"], model_path)

    print(predictions)
