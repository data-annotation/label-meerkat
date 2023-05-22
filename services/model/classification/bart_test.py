import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BartForSequenceClassification, BartTokenizer, AdamW


def convert_label_to_index(labels, label_list):
    str_to_index = {string: index for index, string in enumerate(label_list)}
    index_list = [str_to_index[label] for label in labels]
    return index_list

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label_list: list):
        self.texts = texts
        self.labels = convert_label_to_index(labels, label_list)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        inputs['labels'] = torch.tensor(label)
        return inputs


def train(data: list,
          labels: list,
          batch_size: int = 20,
          num_epochs: int = 20,
          old_model: str = 'facebook/bart-base',
          output_model: str = 'test_model',
          label_list: list = None,
          device='cpu'):
    label_list = label_list or ['positive', 'negative']
    tokenizer = BartTokenizer.from_pretrained(old_model)
    model = BartForSequenceClassification.from_pretrained(old_model, num_labels=len(label_list))
    model.to(device)

    dataset = TextDataset(data, labels, tokenizer, label_list=label_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(num_epochs):
        model.train()
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch + 1}/{num_epochs} completed")

    model.save_pretrained(output_model)
    return output_model


def predict(data: list, model_path: str = 'test_model'):
    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = BartForSequenceClassification.from_pretrained(model_path)
    model.eval()

    inputs = tokenizer(data, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).tolist()

    return predictions


if __name__ == "__main__":
    # 训练数据和标签
    train_data = ['This is a positive sentence.', "I'm feeling great today.", 'I feel so sorry', 'The product is not good.']
    train_labels = ['positive', 'positive', 'negative', 'negative']

    # 训练模型
    model_path = train(train_data, train_labels, num_epochs=5)

    # 预测数据
    test_data = ['I love this product.', 'This is a terrible experience.']

    # 使用训练好的模型进行预测
    predictions = predict(test_data, model_path)

    # 打印预测结果
    print(predictions)
