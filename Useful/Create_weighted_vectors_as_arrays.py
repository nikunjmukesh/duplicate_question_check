# coding: utf-8
# In[2]:
import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer


# In[3]:
print('LOADING WORDNET')
nlp = spacy.load('en_core_web_lg')
print('DONE')

# In[6]:

print('LOADING TEXT')
df = pd.read_csv('./Data/train_cleaned.csv')
print('DONE')
#uncomment for train
df.drop(['id', 'qid1', 'qid2', 'is_duplicate'], axis=1, inplace=True)
#uncomment for test
#df.drop('test_id', axis=1, inplace=True)
df.fillna('NO QUESTION', inplace=True)


# In[7]:


SIZE_OF_TEST = len(list(df['question1']))
#SIZE_OF_TEST = 100

# Calculating TFIDF scores for all

# In[8]:

print('PERFORMING TFIDF ANALYSIS')
questions = list(df['question1']) + list(df['question2'])
tfidf = TfidfVectorizer(lowercase=False,)
tfidf.fit_transform(questions)
word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf.idf_))
print('DONE')

# Calculating vectors for question 1

# In[9]:


w_vectors = []
for i in tqdm(range(SIZE_OF_TEST), desc='WEIGHTED VECTORS FOR QUESTION1'):
    doc = nlp(df['question1'][i])
    mean_vec = np.zeros([len(doc), 300])
    for word in doc:
        vec = word.vector
        try:
            idf = word2tfidf[str(word)]
        except:
            idf = 0
        mean_vec += vec*idf
    mean_vec = mean_vec.mean(axis=0)
    w_vectors.append(mean_vec)
np_vec = np.array(w_vectors)
del w_vectors


# In[8]:

print('SAVING QUESTION 1 VECTORS TO FILE')
np_vec.tofile('./Data/Train/train_q1_vectors')
del np_vec


# In[9]:


w_vectors = []
for i in tqdm(range(SIZE_OF_TEST), desc='WEIGHTED VECTORS FOR QUESTION2'):
    doc = nlp(df['question2'][i])
    mean_vec = np.zeros([len(doc), 300])
    for word in doc:
        vec = word.vector
        try:
            idf = word2tfidf[str(word)]
        except:
            idf = 0
        mean_vec += vec*idf
    mean_vec = mean_vec.mean(axis=0)
    w_vectors.append(mean_vec)
np_vec = np.array(w_vectors)
del w_vectors


# In[10]:

print('SAVING QUESTION 2 VECTORS TO FILE')
np_vec.tofile('./Data/Train/train_q2_vectors')
del np_vec

