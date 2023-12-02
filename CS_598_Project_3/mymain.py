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

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def vectorizer(min_df = None, max_df = None, ngram_range = (1, 2), vector_type = 'tfidf'):
    vector = None
    if vector_type == 'tfidf':
        vector = TfidfVectorizer()
    elif vector_type == 'count':
        vector = CountVectorizer()
    if min_df is not None:
        vector.set_params(min_df=min_df)
    if max_df is not None:
        vector.set_params(max_df=max_df)

    vector.set_params(stop_words=stopwords, ngram_range=ngram_range, preprocessor=lambda x: x.lower(), token_pattern=r"\b[\w+\|']+\b")
    return vector

vector_1 = vectorizer(min_df=0.0001, max_df=0.5, ngram_range=(1, 2))
labels = train['sentiment'].to_numpy()
dtm = vector_1.fit_transform(train['review'])
dtm_positive = dtm[labels == 1]
dtm_negative = dtm[labels == 0]

def compute_t_test(feature_index, dtm_positive, dtm_negative):
    positive_feature_values = dtm_positive[:, feature_index].toarray().flatten()
    negative_feature_values = dtm_negative[:, feature_index].toarray().flatten()

    t_stat, p_value = stats.ttest_ind(positive_feature_values, negative_feature_values, equal_var=True)
    return t_stat, p_value

# Run the t-tests in parallel.
t_test_results = Parallel(n_jobs=-1)(delayed(compute_t_test)(i, dtm_positive, dtm_negative) for i in range(dtm.shape[1]))

# Extract t-statistics and p-values from the results.
t_stats, p_values = zip(*t_test_results)

t_test_results = pd.DataFrame({
    'Feature': vector_1.get_feature_names_out(),
    't-statistic': t_stats,
    'p-value': p_values,
    'abs t-statistic': np.absolute(t_stats)
})

significant_features = t_test_results[t_test_results['p-value'] < 0.05]
significant_features = significant_features.sort_values(by='abs t-statistic', ascending=False)
neative_features = significant_features[significant_features['t-statistic'] < 0]
positive_features = significant_features[significant_features['t-statistic'] > 0]
common_features = pd.merge(neative_features, positive_features, on='Feature')

vector_2 = vectorizer(min_df=0.0001, max_df=0.5, ngram_range=(1, 2))
vector_2.fit(significant_features['Feature'].head(2000))
significant_dmt_train = vector_2.transform(train_review)

# use lassocv to extract features
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_selection import SelectFromModel

model = LogisticRegressionCV(max_iter=10000, cv=5, n_jobs=-1, penalty='l1', solver='liblinear')
model.fit(significant_dmt_train, train_label)

coefficients = model.coef_.flatten()
feature_names = vector_2.get_feature_names_out()
feature_importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
# Calculate the absolute values of coefficients for ranking
feature_importance['Absolute Coefficient'] = feature_importance['Coefficient'].abs()
# Sort the features by absolute coefficient values
sorted_features = feature_importance.sort_values(by='Absolute Coefficient', ascending=False)

vector_3 = vectorizer(min_df=0.0001, max_df=0.5, ngram_range=(1, 2))
vector_3.fit(sorted_features['Feature'].head(1000))
dtm_train = vector_3.transform(train_review)
dtm_test = vector_3.transform(test_review)

model = LogisticRegressionCV(max_iter=10000, cv=5, n_jobs=-1, solver='liblinear')
model.fit(dtm_train, train_label)

from sklearn.metrics import roc_auc_score

pred = model.predict_proba(dtm_test)[:, 1]
print(roc_auc_score(test_label, pred))

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
def score(train_review, train_label, test_review, test_label):
    # fit with logistic regression for classification
    from sklearn.metrics import accuracy_score

    model = LogisticRegressionCV(cv=5, max_iter=10000, n_jobs=-1)
    model.fit(train_review, train_label)
    # calculate AUC score
    pred = model.predict_proba(test_review)

    return roc_auc_score(test_label, pred[:, 1]), pred[:, 1]


train_score, test_score = fit_vectorizer(train_review.copy(), test_review.copy(), vocab)
auc, pred = score(train_score, train_label, test_score, test_label)

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