
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import euclidean, cosine
import math
from collections import Counter


# In[2]:
START_INDEX = 2100000
END_INDEX = 2200000



def pre_process(l1, l2):
    s1 = ' '.join(l1)
    s2 = ' '.join(l2)
    return s1, s2


# In[3]:


def get_vectors(q1, q2):
    s1, s2 = pre_process(q1, q2)
    v = CountVectorizer(stop_words=None, token_pattern=r"(?u)\b\w+\b").fit([s1, s2])
    vec1, vec2 = v.transform([s1, s2])
    v1 = vec1.toarray().ravel()
    v2 = vec2.toarray().ravel()
    return v1, v2


# In[4]:


def normalize(vec):
    n = np.linalg.norm(vec)
    if n == 0:
        return vec
    return vec/n


# In[5]:


def similar(q1, q2):
    v1, v2 = get_vectors(q1, q2)
    cos = cosine(v1, v2)
    if(cos<-1):
        deg = math.degrees(math.acos(-1))
    elif(cos>1):
        deg = math.degrees(math.acos(1))
    else:
        deg = math.degrees(math.acos(cos))
    return euclidean(v1, v2), cos, deg


# In[6]:


def similarities(q1, q2):
    sim_features = [0.0]*3
    sim_features[0], sim_features[1], sim_features[2] = similar(q1, q2)
    return sim_features


# In[7]:


def alpha_similar(q1, q2):
    if(len(set(q1))!=1 | len(set(q2))!=1 ):
        c = sum(1 for j in q1 if j==0) + sum(1 for j in q2 if j==0)
        return c
    else:
        return 0


# In[8]:


def insert_features(df, similar_df):
    n = len(similar_df.columns)
    for i in tqdm(range(len(df['lemma_1'])), desc="GENERATING VALUES"):
        f = np.empty(n) * np.nan
        f[0], f[1], f[2] = similarities(df['lemma_1'][START_INDEX+i], df['lemma_2'][START_INDEX+i])
        f[3], f[4], f[5] = similarities(df['POS_1'][START_INDEX+i], df['POS_2'][START_INDEX+i])
        f[6], f[7], f[8] = similarities(df['dependency_1'][START_INDEX+i], df['dependency_2'][START_INDEX+i])
        f[9] = alpha_similar(df['alpha_1'][START_INDEX+i], df['alpha_2'][START_INDEX+i])
        similar_df.loc[len(similar_df)] = f   


# In[9]:


df_q1 = pd.read_pickle('./Features/Test/Test_nlp_features_1', compression='gzip')
df_q2 = pd.read_pickle('./Features/Test/Test_nlp_features_2', compression='gzip')

df_1 = df_q1[START_INDEX:END_INDEX]
df_2 = df_q2[START_INDEX:END_INDEX]

del df_q1, df_q2
print("FROM {} TO {}".format(START_INDEX, END_INDEX))
# In[10]:


df = pd.DataFrame()
for i in df_1.columns:
    df[i+'_1'] = df_1[i]
    df[i+'_2'] = df_2[i]


# In[11]:


del df_1, df_2


# In[12]:


similar_df = pd.DataFrame(columns=['lemma_euc', 'lemma_cos', 'lemma_deg',
                                   'POS_euc', 'POS_cos', 'POS_deg', 
                                   'dep_euc', 'dep_cos', 'dep_deg',
                                   'alpha_diff'])


# In[13]:


insert_features(df, similar_df)


# In[14]:


similar_df.to_pickle('./Features/Test/Pure_NLP_comparision_{}_{}'.format(START_INDEX, END_INDEX), compression='gzip')


# In[15]:


similar_df.to_csv('./Features/Test/Pure_NLP_comparision.csv')

