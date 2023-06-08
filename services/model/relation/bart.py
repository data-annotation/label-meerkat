from typing import Union
import pandas as pd
import torch
from transformers import BartTokenizer, BartForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from services.model.util.trans_util import label2id, id2label
from services.model import device
from transformers import pipeline

default_model = '../bart-base'

def load_from_pretrained(model_path: str = None,
                         label_list: int = None):
    model_path = model_path or default_model
    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = BartForSequenceClassification.from_pretrained(model_path,
                                                          num_labels=len(label_list),
                                                          id2label={i: l for i, l in enumerate(label_list)}, 
                                                          label2id={l: i for i, l in enumerate(label_list)})
    return model, tokenizer

def train(data, 
          labels, 
          batch_size=20, 
          num_epochs=20, 
          old_model=None,
          output_model='test_model',
          label_list=None,
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
    labels = label2id(labels, label_list=label_list)
    model, tokenizer = load_from_pretrained(old_model, label_list)
    train_dataset = Dataset.from_dict({'text': data, 'label': labels})

    train_dataset = train_dataset.map(lambda example: tokenizer(example['text'],
                                                                add_special_tokens=True,
                                                                padding='max_length',
                                                                truncation=True,
                                                                max_length=512),
                                    batched=True)
    train_dataset.set_format('torch',
                            columns=['input_ids', 'attention_mask', 'label'])

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

    device = trainer.model.device
    if device.type == 'cuda':
        print("##### Trainer is using GPU for training. #####")
    else:
        print("##### Trainer is using CPU for training. #####")

    trainer.train()
    tokenizer.save_pretrained(output_model)
    model.save_pretrained(output_model)
    return output_model


def predict(data: Union[list, pd.DataFrame],
            label_list: list,
            model_path: str = None,
            device: str = device):
    model, tokenizer = load_from_pretrained(model_path, label_list)
    model.to(device)
    model.eval()
    predictions = []
    for text in data:
        encoding = tokenizer.batch_encode_plus(text,  add_special_tokens=True, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions.append(torch.argmax(logits).item())
    return id2label(predictions, label_list)


if __name__ == "__main__":
    # 定义标签列表
    label_list = ['entailment', 'neutral', 'contradiction']

    # 训练数据
    train_data = [
        ['Children smiling and waving at camera', 'The kids are frowning'],
        ['Two blond women are hugging one another.', 'The women are sleeping.'],
        ['A couple play in the tide with their young son.', 'The family is outside.']
    ]
    train_labels = ['entailment', 'neutral', 'contradiction']

    model_path = train(train_data,
                       train_labels,
                       num_epochs=100,
                       label_list=label_list)

    predictions = predict(train_data,
                          label_list=label_list,
                          model_path=model_path)

    print(predictions)
