#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


train = pd.read_csv(r'train.tsv', sep='\t')
test = pd.read_csv(r'test.tsv', sep='\t')
test_y = pd.read_csv(r'test_y.tsv', sep='\t')

train_label = train['sentiment']
train_review =  train['review'].str.replace('&lt;.*?&gt;', ' ', regex=True)

test_label = test_y['sentiment']
test_review = test['review'].str.replace('&lt;.*?&gt;', ' ', regex=True)


# In[3]:


train.head()


# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    stop_words='english',
    lowercase=True,  # Converts all text to lowercase by default
    ngram_range=(1, 4),  # Extracts unigrams only by default
    preprocessor=lambda x: x.lower(),  # Convert to lowercase
    min_df=0.001,                        # Minimum term frequency
    max_df=0.5,                       # Maximum document frequency
    token_pattern=r"\b[\w+\|']+\b" # Use word tokenizer: See Ethan's comment below
)
dtm_train = vectorizer.fit_transform(train['review'])


# In[5]:


dtm_train.shape


# In[6]:


dtm_test = vectorizer.transform(test['review'])


# In[7]:


# use lassocv to extract features
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_selection import SelectFromModel

model = LogisticRegressionCV(max_iter=10000, cv=5, n_jobs=-1, penalty='l1', solver='liblinear')
model.fit(dtm_train, train_label)


# In[8]:


coefficients = model.coef_.flatten()

feature_names = vectorizer.get_feature_names_out()
feature_importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Calculate the absolute values of coefficients for ranking
feature_importance['Absolute Coefficient'] = feature_importance['Coefficient'].abs()

# Sort the features by absolute coefficient values
sorted_features = feature_importance.sort_values(by='Absolute Coefficient', ascending=False)
# sorted_features.head(20)


# In[9]:


# save top 1000 features to myvocab_1000.txt
# save top 2000 features to myvocab_2000.txt
# save top 3000 features to myvocab_3000.txt
vocab_1000 = sorted_features['Feature'].head(1000)
vocab_2000 = sorted_features['Feature'].head(2000)
vocab_3000 = sorted_features['Feature'].head(3000)
vocabs = [vocab_1000, vocab_2000, vocab_3000]


# In[10]:


def read_data():
    train = pd.read_csv(r'train.tsv', sep='\t')
    test = pd.read_csv(r'train.tsv', sep='\t')
    test_y = pd.read_csv(r'train.tsv', sep='\t')

    train_label = train['sentiment']
    train_review =  train['review'].str.replace('&lt;.*?&gt;', ' ', regex=True)

    test_label = test_y['sentiment']
    test_review = test['review'].str.replace('&lt;.*?&gt;', ' ', regex=True)
    
    return train_review, train_label, test_review, test_label
# read myvocab.txt
vocab = pd.read_csv(r'myvocab.txt')


# In[11]:


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


# In[12]:


from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
def score(train_review, train_label, test_review, test_label):
    # fit with logistic regression for classification
    from sklearn.metrics import accuracy_score

    model = LogisticRegressionCV(cv=5, max_iter=10000, n_jobs=-1)
    model.fit(train_review, train_label)
    # calculate AUC score
    pred = model.predict_proba(test_review)

    return roc_auc_score(test_label, pred[:, 1]), pred


# In[13]:


train_review, train_label, test_review, test_label= read_data()


# In[15]:


for vocab in vocabs:
    train_score, test_score = fit_vectorizer(train_review.copy(), test_review.copy(), vocab)
    auc, pred = score(train_score, train_label, test_score, test_label)


# In[16]:


import csv
csv_file_path = r'mysubmission.csv'
test_id = pd.read_csv(r'test.tsv', sep='\t')
ids = test_id['id']
# Write the CSV file
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    # Write the header
    writer.writerow(['id', 'prob'])

    # Write the data
    for id_value, prob_value in zip(ids, pred):
        writer.writerow([id_value, prob_value[0]])




# %%
