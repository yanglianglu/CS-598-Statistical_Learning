import pandas as pd
import numpy as np
from scipy import stats
from joblib import Parallel, delayed
# import nlty and download stopwords
import nltk
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
train = pd.read_csv(r'train.tsv', sep='\t')
test = pd.read_csv(r'test.tsv', sep='\t')
test_y = pd.read_csv(r'test_y.tsv', sep='\t')
train_label = train['sentiment']
train_review =  train['review'].str.replace('&lt;.*?&gt;', ' ', regex=True)
test_label = test_y['sentiment']
test_review = test['review'].str.replace('&lt;.*?&gt;', ' ', regex=True)

vocab = pd.read_csv('myvocab.txt', sep='\t')

def vectorizer(min_df = None, max_df = None, ngram_range = (1, 2)):
    vector = TfidfVectorizer(
        preprocessor=lambda x: x.lower(),  # Convert to lowercase
        stop_words=stopwords,             # Remove stop words
        ngram_range=ngram_range,               # Use 1- to 4-grams
          # Use word tokenizer: See Ethan's comment below
    )
    if min_df is not None:
        vector.set_params(min_df=min_df)
    if max_df is not None:
        vector.set_params(max_df=max_df)
    return vector

from sklearn.feature_extraction.text import TfidfVectorizer

def fit_vectorizer(train_review, test_review, vocab):
    vectorizer = TfidfVectorizer(
    stop_words='english',
    lowercase=True,  # Converts all text to lowercase by default
    ngram_range=(1, 4),  # Extracts unigrams only by default
    preprocessor=lambda x: x.lower(),  # Convert to lowercase
    token_pattern=r"\b[\w+\|']+\b" # Use word tokenizer: See Ethan's comment below
    )
    vectorizer.fit(vocab.values.flatten())
    train_review = vectorizer.transform(train_review)
    test_review = vectorizer.transform(test_review)
    return train_review, test_review

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegressionCV

from sklearn.metrics import accuracy_score

train_score, test_score = fit_vectorizer(train_review.copy(), test_review.copy(), vocab)
model = LogisticRegressionCV(cv=5, max_iter=10000, n_jobs=-1)
model.fit(train_score, train_label)

pred = model.predict_proba(test_score)[:, 1]

import csv
csv_file_path = "mysubmission.csv"
test_id = pd.read_csv('test.tsv', sep='\t')
ids = test_id['id']
# Write the CSV file
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    # Write the header
    writer.writerow(['id', 'prob'])

    # Write the data
    for id_value, prob_value in zip(ids, pred):
        writer.writerow([id_value, prob_value])