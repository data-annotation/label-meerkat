import pandas as pd
from torch import nn
import torch
from transformers import BertTokenizer, BertModel
from typing import List, Dict, Tuple, Type, Union

# Text classification model
class TextClassifier(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super(TextClassifier, self).__init__()
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, embeddings):
        logits = self.fc(embeddings)
        return logits

model_name = 'bert-base-uncased'
model = BertModel.from_pretrained(model_name).to('cuda')
tokenizer = BertTokenizer.from_pretrained(model_name)

# Initialize and train the text classification model
num_classes = 1
hidden_size = 768

classifier = TextClassifier(num_classes, hidden_size).to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)


# Perform inference on new texts
def predict(data: Union[str, List[str]], model_path: str = 'model.pth'):

    # new_texts = ['Some example text', 'Another example text']
    if isinstance(data, str):
        data = [data]

    model = torch.load(model_path)
    model.to('cuda')
    model.eval()

    new_tokens = tokenizer.batch_encode_plus(
        data,
        truncation=True,
        padding=True,
        return_tensors='pt'
    )
    new_input_ids = new_tokens['input_ids'].to('cuda')
    new_attention_mask = new_tokens['attention_mask'].to('cuda')

    with torch.no_grad():
        new_outputs = model(new_input_ids, new_attention_mask)
        new_embeddings = new_outputs.last_hidden_state

        predictions = classifier(new_embeddings)
        predicted_labels = torch.argmax(predictions, dim=1)

    # Print the predicted labels
    for text, label in zip(new_texts, predicted_labels):
        print(f'Text: {text}, Predicted Label: {label.item()}')


def train(texts, labels, batch_size: int = 20, num_epochs: int = 20, model_output_path: str = None):
    
    if isinstance(texts, str):
        texts = [texts]

    # Create a DataFrame
    df = pd.DataFrame({'text': texts, 'label': labels})

    # Save DataFrame to CSV file
    df.to_csv('data.csv', index=False)

    # Load and preprocess the CSV file
    data = pd.read_csv('data.csv')
    texts = data['text'].tolist()
    labels = data['text'].tolist()

    # Tokenize the texts
    tokens = tokenizer.batch_encode_plus(
        texts,
        truncation=True,
        padding=True,
        return_tensors='pt'
    )

    # Generate the embeddings
    input_ids = tokens['input_ids'].to('cuda')
    attention_mask = tokens['attention_mask'].to('cuda')
    
    model.train()

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        embeddings = outputs.last_hidden_state
    
    # Training loop
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

            if (step + 1) % 1 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{total_steps}], Loss: {loss.item()}')
    
    torch.save(model, 'model.pth')

texts = [
    "This is a positive sentence.",
    "I'm feeling great today.",
    "Negative feedback received.",
    "The product is not good.",
]

labels = [1, 1, 0, 0]  # 1 for positive, 0 for negative

train(texts, labels)


new_texts = ['Some example text', 'Another example text']

predict(new_texts)