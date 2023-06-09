import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import joblib
import os

from services.model.util.trans_util import id2label


def train(data,
          labels,
          batch_size=20,
          num_epochs=20,
          old_model='path/to/old/model',
          output_model='test_model',
          label_list=None,
          device='cpu'):
    # 将数据转换为 DataFrame 格式
    if isinstance(data, list):
        data = pd.DataFrame(data, columns=['text'])
    elif isinstance(data, pd.DataFrame):
        data = data.copy()

    # 训练集和验证集划分
    X_train, X_val, y_train, y_val = train_test_split(data['text'], labels, test_size=0.2, random_state=42)

    # 如果存在上一次训练好的模型，加载模型
    if old_model:
        model = joblib.load(old_model)
    else:
        # 创建一个基于 TF-IDF 的SVM模型
        model = make_pipeline(TfidfVectorizer(), SVC())

    # 训练模型
    for epoch in range(num_epochs):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        print(f"Epoch {epoch + 1}/{num_epochs}")

    os.makedirs(output_model, exist_ok=True)
    output_model = '/'.join([output_model, 'svm_model'])
    # 保存训练好的模型
    joblib.dump(model, output_model)
    return output_model


def predict(data, label_list, model_path='path/to/old/model', device='cpu'):
    # 将数据转换为 DataFrame 格式
    if isinstance(data, list):
        data = pd.DataFrame(data, columns=['text'])
    elif isinstance(data, pd.DataFrame):
        data = data.copy()

    model_path = '/'.join([model_path, 'svm_model'])
    # 加载训练好的模型
    model = joblib.load(model_path)

    # 对数据进行预测
    y_pred = model.predict(data['text'])

    return id2label(y_pred.tolist(), label_list)


if __name__ == "__main__":
    data = ['This is a positive sentence.', 'I\'m feeling great today.', 'Negative feedback received.',
            'The product is not good.']
    labels = ['positive', 'positive', 'negative', 'positive']

    model_path = train(data,
                       labels,
                       num_epochs=1,
                       old_model=None,
                       label_list=['positive', 'negative'])

    predictions = predict(['I feel good', "it's so bad"],
                          model_path='test_model',
                          label_list=['positive', 'negative'])

    print(predictions)
