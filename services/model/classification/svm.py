import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix
import pickle

train_set = pd.read_csv('data.csv')


def process_data(data: pd.DataFrame):

    cleanedData = []
    lemma = WordNetLemmatizer()
    swords = stopwords.words("english")

    for text in data['text']:
        # Tokenizing and lemmatizing
        text = nltk.word_tokenize(text.lower())
        text = [lemma.lemmatize(word) for word in text]
    
        # Removing stopwords
        text = [word for word in text if word not in swords]
    
        # Joining
        text = " ".join(text)  
        cleanedData.append(text)

    return cleanedData


def train(dataframe: pd.DataFrame):
    data = process_data(dataframe)
    vectorizer = CountVectorizer(max_features=10000)
    data_transform = vectorizer.fit_transform(data)
    model = SVC()
    model.fit(data_transform,
              np.array(dataframe["label"]))
    
    with open('svm/svm.pickle', 'wb') as fw:
        pickle.dump(model, fw)

    with open('svm/count_vect.pickle', 'wb') as fw:
        pickle.dump(vectorizer, fw)

    
def predict(dataframe: pd.DataFrame):
    with open('svm/svm.pickle', 'rb') as svm:
        model = pickle.load(svm)

    with open('svm/count_vect.pickle', 'rb') as count_vect:
        count_vect = pickle.load(count_vect)

    data = process_data(dataframe)
    data_transform = count_vect.fit_transform(data)
    predictions = model.predict(data_transform)
# predictions = model.predict(x_test)
    print("Accuracy of model is {}%".format(accuracy_score(dataframe['label'].to_numpy(),predictions) * 100))


if __name__ == '__main__':
    # train(train_set)
    predict(train_set)