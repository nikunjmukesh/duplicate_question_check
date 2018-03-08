# coding: utf-8
# In[ ]:
import numpy as np
import pandas as pd
import spacy
import cython
import math
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import euclidean, cosine, canberra, correlation


# In[ ]:


sem_list = ['euclidean', 'cosine', 'cosine_angle', 'canberra', 'correlation']
df_sem = pd.DataFrame(columns=sem_list)


# In[ ]:


START_INDEX = 2100000
END_INDEX = 2400000
NUMBER_OF_FEATURES = len(df_sem.columns)


# In[ ]:


print('LOADING TRAINING DATA '+str(START_INDEX) +' ' + str(END_INDEX))


# In[ ]:
test = pd.read_csv("./Data/train_cleaned.csv")
test.fillna('memento', inplace=True)
train_data = pd.read_csv("./Data/test_cleaned.csv")
train_data.fillna('memento', inplace=True)
df = train_data[START_INDEX:] 
#tdf = pd.read_csv("./Data/train_cleaned.csv")
df.fillna('NO QUESTION', inplace=True)
print('LOADING DONE ')#+str(START_INDEX))
tf = pd.concat([train_data, test])
del train_data
del test


# In[ ]:


print('LOADING WORDNET')


# In[ ]:


nlp = spacy.load('en_core_web_lg')
print('WORDNET LOADED')


# In[ ]:


print("PERFORMING TFIDF ANALYSIS")


# In[ ]:


questions = list(tf['question1']) + list(tf['question2'])
del tf
tfidf = TfidfVectorizer(lowercase=False,)
tfidf.fit_transform(questions)
word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf.idf_))
print('TFIDF ANALYSIS DONE')


# In[ ]:


def normalize(vec):
    n = np.linalg.norm(vec)
    if n == 0:
        return vec
    return vec/n


# In[ ]:


def sent2vec(q1, q2):
    doc_1 = nlp(q1)
    doc_2 = nlp(q2)
    m_1 = np.zeros([len(doc_1), 300])
    m_2 = np.zeros([len(doc_2), 300])
    for word in doc_1:
        vec = word.vector
        try:
            idf = word2tfidf[str(word)]
        except:
            idf = 1
        m_1 += vec * idf
    m_1 = m_1.mean(axis=0)
    for word in doc_2:
        vec = word.vector
        try:
            idf = word2tfidf[str(word)]
        except:
            idf = 1
        m_2 += vec * idf
    m_2 = m_2.mean(axis=0)
    
    return normalize(m_1), normalize(m_2)


# In[ ]:


def similar(q1, q2):
    v1, v2 = sent2vec(q1, q2)
    cos = cosine(v1, v2)
    if(cos<-1):
        deg = math.degrees(math.acos(-1))
    elif(cos>1):
        deg = math.degrees(math.acos(1))
    else:
        deg = math.degrees(math.acos(cos))
    return euclidean(v1, v2), cos, deg, canberra(v1, v2), correlation(v1, v2)


# In[ ]:


for i in tqdm(range(START_INDEX, START_INDEX+len(df['question1'])), desc='CREATING SEMANTIC FEATURES'):
    feature = np.empty(NUMBER_OF_FEATURES) * np.nan
    feature[0], feature[1], feature[2], feature[3], feature[4]= similar(df['question1'][i], df['question2'][i])
    df_sem.loc[len(df_sem)] = feature

print('SAVING PICKLE')
df_sem.to_pickle('./Features/Test/Semantic_features{}'.format(END_INDEX), compression='gzip')
print('SAVING CSV')
df_sem.to_csv('./Features/Test/Semantic_features{}.csv'.format(END_INDEX), index=False)
