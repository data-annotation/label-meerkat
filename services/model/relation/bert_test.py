import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from sklearn.metrics import accuracy_score
from datasets import Dataset
from services.model.util.trans_util import label2id, id2label
from services.model import device
import pandas as pd

class BERTClassifier:
    def __init__(self, label_list, device='cpu'):
        self.label_list = label_list
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.to(device)

    def train(self, data, labels, batch_size=20, num_epochs=20, old_model='path/to/old/model',
              output_model='test_model',label_list=None,
          device=device):
        """
        模型训练方法

        Args:
            data: 要进行训练的句子列表或Pandas DataFrame
            labels: data 中每个句子之间关系所对应的的标签，如['entailment', 'neutral', 'contradiction']
            batch_size: 批量大小
            num_epochs: 训练的轮数
            old_model: 上一次训练好的模型所保存的路径
            output_model: 训练完成后保存的模型路径

        Returns:
            训练完成后保存的模型路径
        """
        if isinstance(data, pd.DataFrame):
            data = data['sentence'].tolist()

        # 特征提取和标签转换
        inputs = self.tokenizer(data, padding=True, truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        labels = torch.tensor(labels).to(self.device)

        # 定义模型
        model = nn.Linear(self.model.config.hidden_size, len(self.label_list))
        model.to(self.device)

        # 定义优化器和损失函数
        optimizer = torch.optim.AdamW(model.parameters())
        criterion = nn.CrossEntropyLoss()

        # 训练模型
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            logits = model(pooled_output)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

        # 保存模型
        model_path = output_model + '.pth'
        torch.save(model.state_dict(), model_path)

        return model_path

    def predict(self, data, model_path='path/to/old/model'):
        """
        使用训练好的模型进行预测

        Args:
            data: 要预测的句子列表或Pandas DataFrame，list of str
            model_path: 使用该模型对data进行预测

        Returns:
            对data进行预测后的label列表
        """
        if isinstance(data, pd.DataFrame):
            data = data['sentence'].tolist()

        # 加载模型
        model = nn.Linear(self.model.config.hidden_size, len(self.label_list))
        model.load_state_dict(torch.load(model_path))
        model.to(self.device)
        model.eval()

        # 特征提取和标签转
