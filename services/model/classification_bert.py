import pandas as pd
from torch import nn
import torch
from transformers import BertTokenizer, BertModel

# Sample data
texts = [
    "This is a positive sentence.",
    "I'm feeling great today.",
    "Negative feedback received.",
    "The product is not good.",
]

labels = [1, 1, 0, 0]  # 1 for positive, 0 for negative

# Create a DataFrame
df = pd.DataFrame({'text': texts, 'label': labels})

# Save DataFrame to CSV file
df.to_csv('data.csv', index=False)

# Load and preprocess the CSV file
data = pd.read_csv('data.csv')
texts = data['text'].tolist()
labels = data['text'].tolist()

model_name = 'bert-base-uncased'
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Tokenize the texts
tokens = tokenizer.batch_encode_plus(
    texts,
    truncation=True,
    padding=True,
    return_tensors='pt'
)

# Generate the embeddings
input_ids = tokens['input_ids']
attention_mask = tokens['attention_mask']

with torch.no_grad():
    outputs = model(input_ids, attention_mask)
    embeddings = outputs.last_hidden_state


# Text classification model
class TextClassifier(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super(TextClassifier, self).__init__()
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, embeddings):
        logits = self.fc(embeddings)
        return logits

# Initialize and train the text classification model
num_classes = 1
hidden_size = 768

classifier = TextClassifier(num_classes, hidden_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

# Training loop
num_epochs = 10
batch_size = 32
total_steps = len(embeddings) // batch_size

for epoch in range(num_epochs):
    for step in range(total_steps):
        start = step * batch_size
        end = (step + 1) * batch_size

        inputs = embeddings[start:end]
        targets = torch.tensor(labels[start:end])

        logits = classifier(inputs)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{total_steps}], Loss: {loss.item()}')

# Perform inference on new texts
new_texts = ['Some example text', 'Another example text']
new_tokens = tokenizer.batch_encode_plus(
    new_texts,
    truncation=True,
    padding=True,
    return_tensors='pt'
)
new_input_ids = new_tokens['input_ids']
new_attention_mask = new_tokens['attention_mask']

with torch.no_grad():
    new_outputs = model(new_input_ids, new_attention_mask)
    new_embeddings = new_outputs.last_hidden_state

    predictions = classifier(new_embeddings)
    predicted_labels = torch.argmax(predictions, dim=1)

# Print the predicted labels
for text, label in zip(new_texts, predicted_labels):
    print(f'Text: {text}, Predicted Label: {label.item()}')


# - 针对每个模型+任务 封装出两个方法 一个训练方法，一个预测方法
# - 训练方法参数有「训练数据」「超参数参数（batch_size, epoch等）」「模型训练完成后保存路径」
# - 预测方法参数有「预测数据」「使用的模型路径」，返回 预测结果

def train(data: [pd.DataFrame, dict, list, torch.Tensor],
          batch_size: int = 20,
          epoch: int = 20,
          model_output_path: str = None,
          **kwargs):
    pass


def predict(data: list,
            model_output_path: str = 'path/to/model',
            **kwargs):
    model = BertModel.from_pretrained(model_output_path)
    new_tokens = tokenizer.batch_encode_plus(
        new_texts,
        truncation=True,
        padding=True,
        return_tensors='pt'
    )

    new_input_ids = new_tokens['input_ids']
    new_attention_mask = new_tokens['attention_mask']
    new_outputs = model(new_input_ids, new_attention_mask)
    new_embeddings = new_outputs.last_hidden_state
    predictions = classifier(new_embeddings)
    predicted_labels = torch.argmax(predictions, dim=1)

    return zip(data, predicted_labels)


