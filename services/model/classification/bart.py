from transformers import AutoTokenizer, BartModel
import pandas as pd
from torch import nn
import torch
from typing import List, Dict, Tuple, Type, Union
from services.model import device


# Text classification model
class TextClassifier(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super(TextClassifier, self).__init__()
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, embeddings):
        logits = self.fc(embeddings)
        return logits

def load_model(model_id: str):
    # Load the pre-trained BART model and tokenizer
    model_id = model_id or 'facebook/bart-base'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = BartModel.from_pretrained(model_id).to(device)
    return model, tokenizer

def get_classifier(num_classes: str=1, hidden_size: int = 768):
    # Initialize and train the text classification model
    classifier = TextClassifier(num_classes, hidden_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    return classifier, criterion, optimizer


# Perform inference on new texts
def predict(data: list, model_path: str):
    model, tokenizer = load_model(model_path)
    model.eval()
    classifier, criterion, optimizer = get_classifier()
    # Tokenize and encode the input texts
    input_ids = []
    attention_masks = []
    for text in data:
        encoding = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        input_ids.append(encoding['input_ids'].squeeze())
        attention_masks.append(encoding['attention_mask'].squeeze())

    # Convert lists to tensors
    input_ids = torch.stack(input_ids).to(device)
    attention_masks = torch.stack(attention_masks).to(device)

    # Forward pass
    with torch.no_grad():
        embeddings = model(input_ids=input_ids, attention_mask=attention_masks).last_hidden_state
        print(embeddings.shape)
        predictions = classifier(embeddings)
        predicted_labels = torch.argmax(predictions, dim=1)

    # Print the predicted labels
    # for text, label in zip(new_texts, predicted_labels):
    #     print(f'Text: {text}, Predicted Label: {label.item()}')
    return predicted_labels


def train(texts: list,
          labels: list,
          batch_size: int = 20,
          num_epochs: int = 20,
          old_model: str = None,
          new_model: str = 'test_model'):
    model, tokenizer= load_model(old_model)
    model.train()
    classifier, criterion, optimizer = get_classifier()
    # Tokenize and encode the input texts
    input_ids = []
    attention_masks = []
    for text in texts:
        encoding = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        input_ids.append(encoding['input_ids'].squeeze())
        attention_masks.append(encoding['attention_mask'].squeeze())

    # Convert lists to tensors
    input_ids = torch.stack(input_ids).to(device)
    attention_masks = torch.stack(attention_masks).to(device)

    # Forward pass
    with torch.no_grad():
        embeddings = model(input_ids=input_ids, attention_mask=attention_masks).last_hidden_state
        print('*'*30)
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
    
    torch.save(model, new_model)

if __name__ == "__main__":
    texts = [
        "This is a positive sentence.",
        "I'm feeling great today.",
        "Negative feedback received.",
        "The product is not good.",
    ]

    labels = [1, 1, 0, 0]  # 1 for positive, 0 for negative

    train(texts, labels)


    new_texts = ['Some example text', 'Another example text']

    # predict(new_texts, 'test_model')
