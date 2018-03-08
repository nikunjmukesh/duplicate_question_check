# coding: utf-8
# In[1]:
import gensim
import cython
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split


# In[2]:
google_vector = gensim.models.KeyedVectors.load_word2vec_format('./Data/GoogleNews-vectors-negative300.bin.gz', binary=True)


# In[3]:
df = pd.read_csv('./Data/Cleaned_train_lac.csv')
df = df.drop(['id', 'qid1', 'qid2', 'is_duplicate'], axis=1)
df.fillna('NO QUESTION', inplace = True)


# In[5]:
from tqdm import tqdm
model_WS = {'question1':[], 'question2':[]}
for i in tqdm(range(len(df['question1'])), desc='GENERATING WORD VECTORS'):
    model_WS['question1'].append(gensim.models.Word2Vec([df['question1'][i].split()], size=300, window=3, min_count=1, workers=4))
    model_WS['question2'].append(gensim.models.Word2Vec([df['question2'][i].split()], size=300, window=3, min_count=1, workers=4))
print('---------------DONE----------------')


# In[6]:
for i in tqdm(range(len(model_WS['question1'])), desc='SAVING WORD VECTORS'):
    model_WS['question1'][i].wv.save('./Models/question1/{}.csv'.format(i))
    model_WS['question2'][i].wv.save('./Models/question2/{}.csv'.format(i))
print('---------------DONE----------------')


# In[8]:
model_WS_sentences = {'question1':[], 'question2':[]}
for i in tqdm(range(len(model_WS['question1'])), desc='CALCULATING SENTENCE VECTORS'):
    sent_vec_1 = np.zeros([300])
    sent_vec_2 = np.zeros([300])
    for j in model_WS['question1'][i].wv.vocab:
        sent_vec_1 = sent_vec_1 + model_WS['question1'][i][j]
    for j in model_WS['question2'][i].wv.vocab:
        sent_vec_2 = sent_vec_2 + model_WS['question2'][i][j]
    sent_vec_1 = sent_vec_1/len(df['question1'][i])
    sent_vec_2 = sent_vec_2/len(df['question2'][i])
    model_WS_sentences['question1'].append(sent_vec_1)
    model_WS_sentences['question2'].append(sent_vec_2)  
print('---------------DONE----------------')


# In[9]:
df_sen = pd.DataFrame(columns=['question1', 'question2'])
df_sen['question1'] = model_WS_sentences['question1']
df_sen['question2'] = model_WS_sentences['question2']


# In[12]:
df_sen.to_pickle('./Models/sentences/__Sent2vec_non_weighted.csv', compression='gzip')

