from transformers import AutoTokenizer, GPT2ForTokenClassification
from transformers import pipeline
import pandas as pd
import torch
from typing import Union
from torch.utils.data import Dataset, DataLoader
from torch import cuda
from sklearn.metrics import accuracy_score
import os

data = pd.read_csv("clean_data.csv")

label2id = {'O': 0,
            'B-geo': 1,
            'B-gpe': 2,
            'B-per': 3,
            'I-geo': 4,
            'B-org': 5,
            'I-org': 6,
            'B-tim': 7,
            'I-per': 8,
            'I-gpe': 9,
            'I-tim': 10}

id2label = dict(zip(label2id.values(), label2id.keys()))


def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    """
    Word piece tokenization makes it difficult to match word labels
    back up with individual word pieces. This function tokenizes each
    word one at a time so that it is easier to preserve the correct
    label for each subword. It is, of course, a bit slower in processing
    time, but it will help our model achieve higher accuracy.
    """

    tokenized_sentence = []
    labels = []

    sentence = sentence.strip()

    for word, label in zip(sentence.split(), text_labels.split(",")):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels


class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        # step 1: tokenize (and adapt corresponding labels)
        sentence = self.data.sentence[index]
        word_labels = self.data.word_labels[index]
        tokenized_sentence, labels = tokenize_and_preserve_labels(
            sentence, word_labels, self.tokenizer)

        # step 2: add special tokens (and corresponding labels)
        tokenized_sentence = ["[CLS]"] + \
            tokenized_sentence + ["[SEP]"]  # add special tokens
        labels.insert(0, "O")  # add outside label for [CLS] token
        labels.insert(-1, "O")  # add outside label for [SEP] token

        # step 3: truncating/padding
        maxlen = self.max_len

        if (len(tokenized_sentence) > maxlen):
            # truncate
            tokenized_sentence = tokenized_sentence[:maxlen]
            labels = labels[:maxlen]
        else:
            # pad
            tokenized_sentence = tokenized_sentence + \
                ['[PAD]'for _ in range(maxlen - len(tokenized_sentence))]
            labels = labels + ["O" for _ in range(maxlen - len(labels))]

        # step 4: obtain the attention mask
        attn_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_sentence]

        # step 5: convert tokens to input ids
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)

        label_ids = [label2id[label] for label in labels]
        # the following line is deprecated
        #label_ids = [label if label != 0 else -100 for label in label_ids]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(attn_mask, dtype=torch.long),
            # 'token_type_ids': torch.tensor(token_ids, dtype=torch.long),
            'targets': torch.tensor(label_ids, dtype=torch.long)
        }

    def __len__(self):
        return self.len

# define params


MAX_LEN = 128
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 1e-05
MAX_GRAD_NORM = 10
tokenizer = AutoTokenizer.from_pretrained('gpt2')


def load_data(data: pd.DataFrame, batch_size: int):
    # divide data to train_set and test_set
    train_size = 0.8
    train_dataset = data.sample(frac=train_size, random_state=200)
    test_dataset = data.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)
    training_set = dataset(train_dataset, tokenizer, MAX_LEN)
    testing_set = dataset(test_dataset, tokenizer, MAX_LEN)

    # load dataloader
    train_params = {'batch_size': batch_size,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': batch_size,
                   'shuffle': True,
                   'num_workers': 0
                   }
    print('-----')
    print(len(training_set))

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)
    return training_loader, testing_loader


def train(data: Union[pd.DataFrame, dict, list, torch.Tensor],
          batch_size: int = 20,
          epoch: int = 20,
          model_output_path: str = None,
          **kwargs):
    training_loader, testing_loader = load_data(data, batch_size)

    if os.path.exists(model_output_path):
        model = GPT2ForTokenClassification.from_pretrained(model_output_path, num_labels=len(id2label),
                                                   id2label=id2label,
                                                   label2id=label2id)
        
    else:
        model = GPT2ForTokenClassification.from_pretrained('gpt2', num_labels=len(id2label),
                                                   id2label=id2label,
                                                   label2id=label2id)


    # define optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    # put model in training mode
    model.train()

    # device cuda
    device = 'cuda:0' if cuda.is_available() else 'cpu'
    model.to(device)
    
    for i in range(epoch):
        
        print(f"Training epoch: {i + 1}")

        for idx, batch in enumerate(training_loader):
            ids = batch['ids'].to(device, dtype=torch.long)
            mask = batch['mask'].to(device, dtype=torch.long)
            targets = batch['targets'].to(device, dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
            loss, tr_logits = outputs.loss, outputs.logits
            tr_loss += loss.item()

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if idx % 100 == 0:
                loss_step = tr_loss/nb_tr_steps
                print(f"Training loss per 100 training steps: {loss_step}")

            # compute training accuracy
            flattened_targets = targets.view(-1)  # shape (batch_size * seq_len,)
            # shape (batch_size * seq_len, num_labels)
            active_logits = tr_logits.view(-1, model.num_labels)
            flattened_predictions = torch.argmax(
                active_logits, axis=1)  # shape (batch_size * seq_len,)
            # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
            # active accuracy is also of shape (batch_size * seq_len,)
            active_accuracy = mask.view(-1) == 1
            targets = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(
                flattened_predictions, active_accuracy)

            tr_preds.extend(predictions)
            tr_labels.extend(targets)

            tmp_tr_accuracy = accuracy_score(
                targets.cpu().numpy(), predictions.cpu().numpy())
            tr_accuracy += tmp_tr_accuracy

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=MAX_GRAD_NORM
            )

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = tr_loss / nb_tr_steps
        tr_accuracy = tr_accuracy / nb_tr_steps
        print(f"Training loss epoch: {epoch_loss}")
        print(f"Training accuracy epoch: {tr_accuracy}")
        tr_loss, tr_accuracy = 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0

    model.save_pretrained(model_output_path)


def predict(data: list, model_path: str):
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = GPT2ForTokenClassification.from_pretrained(model_path)

    ner_pipeline = pipeline('ner', tokenizer=tokenizer, model=model)
    predict_label = ner_pipeline(data)
    return zip(data, predict_label)