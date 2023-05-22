import torch
from transformers import BartTokenizer, BartForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset


class SentenceClassifier:
    def __init__(self, old_model='facebook/bart-base', num_labels=2, label_list: list=None):
        label_list = label_list or ["positive", "negative"]
        self.id2label = {i: l for i, l in enumerate(label_list)}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.tokenizer = BartTokenizer.from_pretrained(old_model)
        self.model = BartForSequenceClassification.from_pretrained(old_model,
                                                                   num_labels=num_labels)

    def train(self, data, labels,
              batch_size=10, num_epochs=5,
              old_model='facebook/bart-base',
              output_model='test_model', device='cpu'):
        labels = [self.label2id[i] for i in labels]
        train_dataset = Dataset.from_dict({'text': data, 'label': labels})
        train_dataset = train_dataset.map(lambda example: self.tokenizer(example['text'],
                                                                         padding='max_length',
                                                                         truncation=True,
                                                                         max_length=512),
                                          batched=True)
        train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

        training_args = TrainingArguments(
            output_dir=output_model,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            logging_dir='./logs',
            logging_steps=10,
            disable_tqdm=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset
        )

        trainer.train()

        self.model.save_pretrained(output_model)
        return output_model

    def predict(self, data, model_path='test_model', device='cpu'):
        self.model = BartForSequenceClassification.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

        predictions = []
        for text in data:
            encoding = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions.append(torch.argmax(logits).item())

        return predictions


if __name__ == "__main__":
    data = ['This is a positive sentence.', 'I\'m feeling great today.', 'Negative feedback received.',
            'The product is not good.']
    labels = ['positive', 'positive', 'negative', 'positive']

    classifier = SentenceClassifier()
    model_path = classifier.train(data, labels)

    predictions = classifier.predict(['I feel good', "it's so bad"], model_path)

    print(predictions)
