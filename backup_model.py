# Model for our project during the workshop
# Necessary Imports
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords

# sklearn models, vectorizers, and pipelines
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

from sklearn.pipeline import make_pipeline

# sklearn accuracy measurements and displays
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sns


# Make training testing split
import random

def split_assigner():
    return random.choice([1,2,3,4,5]) # 5 = testing

def create_split(dataset):
    dataset['split'] =  dataset['tweet'].map(lambda x: split_assigner())
    return dataset

def get_split(dataset):
    return {'train': [dataset[dataset['split'] < 5].tweet, dataset[dataset['split'] < 5].author],
    'test': [dataset[dataset['split'] == 5].tweet, dataset[dataset['split'] == 5].author]}

if __name__ == '__main__':
    dataset = pd.read_pickle('train_parser/new.pkl')

    l1nbmodel = make_pipeline(TfidfVectorizer(max_features=300, tokenizer=nltk.word_tokenize, stop_words=stopwords.words('french'), ngram_range=(1,3)), 
    SVC(kernel='rbf', C=1E8))

    # Train the model
    l1nbmodel.fit(dataset.tweet, dataset.author)

    # Load in the test data
    test_df = pd.read_csv('test_data_unlabeled.tsv', sep='\t', names=['tweet'])
    preds = []
    print(test_df.info())
    for tweet in test_df['tweet']:
       preds.append(l1nbmodel.predict([tweet])[0])

    test_df.insert(0, 'predicted_author', preds)

    print(test_df)

    with open('answers.txt', 'w') as src:
        for author in test_df['predicted_author']:
            src.write(author+'\n')