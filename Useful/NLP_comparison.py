# coding: utf-8
# In[87]:
import pandas as pd
from tqdm import tqdm
from collections import Counter

# In[88]:
df_1 = pd.read_pickle('./Data/nlp_features_1', compression='gzip')
df_2 = pd.read_pickle('./Data/nlp_features_2', compression='gzip')


# In[89]:
similar_df = pd.DataFrame(columns=['lemma', 'POS', 'dependency', 'alpha'])


# In[90]:
sim_lemma = []
sim_pos = []
sim_dep = []
sim_alpha = []


# In[91]:
def lemma_similar():
    for i in tqdm(range(len(df_1['lemma'])), desc='CALCULATING LEMMA SIMILARITIES'):
        if len(df_2['lemma'][i]) >= len(df_1['lemma'][i]):
            c = len(list((Counter(df_1['lemma'][i]) & Counter(df_2['lemma'][i])).elements()))
            sim_lemma.append(c/(len(df_1['lemma'][i])+len(df_2['lemma'][i])))
        else:
            c = len(list((Counter(df_2['lemma'][i]) & Counter(df_1['lemma'][i])).elements()))
            sim_lemma.append(c/(len(df_1['lemma'][i])+len(df_2['lemma'][i])))


# In[92]:
def pos_similar():
    for i in tqdm(range(len(df_1['POS'])), desc='CALCULATING POS SIMILARITIES'):
        if len(df_2['POS'][i]) >= len(df_1['POS'][i]):
            c = len(list((Counter(df_1['POS'][i]) & Counter(df_2['POS'][i])).elements()))
            sim_pos.append(c/(len(df_1['POS'][i])+len(df_2['POS'][i])))
        else:
            c = len(list((Counter(df_2['POS'][i]) & Counter(df_1['POS'][i])).elements()))
            sim_pos.append(c/(len(df_1['POS'][i])+len(df_2['POS'][i])))


# In[93]:
def dep_similar():
    for i in tqdm(range(len(df_1['dependency'])), desc='CALCULATING DEPENDENCY SIMILARITIES'):
        if len(df_2['dependency'][i]) >= len(df_1['dependency'][i]):
            c = len(list((Counter(df_1['dependency'][i]) & Counter(df_2['dependency'][i])).elements()))
            sim_dep.append(c/(len(df_1['dependency'][i])+len(df_2['dependency'][i])))
        else:
            c = len(list((Counter(df_2['dependency'][i]) & Counter(df_1['dependency'][i])).elements()))
            sim_dep.append(c/(len(df_1['dependency'][i])+len(df_2['dependency'][i])))


# In[94]:
def alpha_similar():
    for i in tqdm(range(len(df_1['alpha'])), desc='CALCULATING ALPHABET SIMILARITIES'):
        if(len(set(df_1['alpha'][i]))!=1 | len(set(df_1['alpha'][i]))!=1 ):
            c = sum(1 for j in df_1['alpha'][i] if j==0) + sum(1 for j in df_2['alpha'][i] if j==0)
            sim_alpha.append(c/(len(df_1['alpha'][i]) + len(df_2['alpha'][i])))
        else:
            sim_alpha.append(0) 


# In[95]:
lemma_similar()
pos_similar()
dep_similar()
alpha_similar()


# In[96]:
similar_df['lemma'] = sim_lemma
similar_df['POS'] = sim_pos
similar_df['dependency'] = sim_dep
similar_df['alpha'] = sim_alpha


# In[97]:
similar_df.to_pickle('NLP_comparison', compression='gzip')

