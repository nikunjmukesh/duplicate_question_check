# coding: utf-8
# In[1]:
import pandas as pd
import numpy as np
from tqdm import tnrange, tqdm
from IPython.display import display, HTML

#glove_model = gensim.models.KeyedVectors.load_word2vec_format('./Data/100/glove_model.bin', binary=True)
# In[2]:
print('LOADING STAT ANALYSIS INFORMATION')
stat_df = pd.read_pickle('./Data/Complete/Statistical_features', compression='gzip')

# In[3]:
print('LOADING SENTENCE VECTORS')
spacy_df = pd.read_pickle('./Data/Complete/spacy_features_normalized_idf', compression='gzip')

# In[4]:
#s2v_df = pd.read_pickle('./Data/Complete/Sent2vec_non_weighted', compression='gzip')

# In[5]:
print('LOADING NLP COMPARISON INFORMATION')
nlp_df = pd.read_pickle('./Data/Complete/NLP_comparison', compression='gzip')

# In[6]:
print('LOADING NER INFORMATION')
ner_df = pd.read_pickle('./Data/Complete/NER_tags', compression='gzip')

# In[7]:
print('LOADING IS_DUPLICATE INFORMATION')
df = pd.read_csv('./Data/train_cleaned.csv', sep=',')
#df = train_df[:100]
df.drop(['id', 'qid1', 'qid2', 'question1', 'question2'], axis=1)

# In[8]:
f_col_list = []

# In[9]:
for i in list(stat_df):
    f_col_list.append(i)
f_col_list.append('ner_dist')
f_col_list.append('lemma_dist')
f_col_list.append('POS_dist')
f_col_list.append('dep_dist')
f_col_list.append('alpha_dist')
'''
for i in range(300):
    f_col_list.append('q1_s2v_unweighted_{}'.format(i))
for i in range(300):
    f_col_list.append('q2_s2v_unweighted_{}'.format(i))
for i in range(300):
    f_col_list.append('q1_wiki_s2v_unweighted_{}'.format(i))
for i in range(300):
    f_col_list.append('q2_wiki_s2v_unweighted_{}'.format(i))
'''
for i in range(300):
    f_col_list.append('q1_vector_{}'.format(i))
for i in range(300):
    f_col_list.append('q2_vector_{}'.format(i))
f_col_list.append('is_duplicate')

# In[10]:
feature_df = pd.DataFrame(columns=f_col_list)

# In[11]:
feature_df['ner_dist'] = ner_df['compare']
feature_df['lemma_dist'] = nlp_df['lemma']
feature_df['POS_dist'] = nlp_df['POS']
feature_df['dep_dist'] = nlp_df['dependency']
feature_df['alpha_dist'] = nlp_df['alpha']

# In[12]:
for i in list(stat_df):
    feature_df[i] = stat_df[i]

# In[13]:
col_list = ['q1_vector', 'q2_vector']
temp_df = pd.DataFrame(columns=col_list)

# In[14]:
#temp_df['q1_s2v_unweighted'] = s2v_df['question1']
#temp_df['q2_s2v_unweighted'] = s2v_df['question2']
#temp_df['q1_wiki_s2v_unweighted'] = spacy_df['q1_f']
#temp_df['q2_wiki_s2v_unweighted'] = spacy_df['q2_f']
temp_df['q1_vector'] = spacy_df['q1_f_weighted']
temp_df['q2_vector'] = spacy_df['q2_f_weighted']
#del s2v_df
del spacy_df

# In[15]:
for i in tqdm(col_list, desc='Separating vector information', leave=True, ncols=100):
    for k in range(300):#, desc='FOR '+str(col_list[i]), leave=True):
        l_list = []
        for j in range(len(temp_df['q1_vector'])):
            l_list.append(temp_df[i][j][k])
        feature_df[str(i)+'_{}'.format(k)] = l_list

# In[16]:
feature_df['is_duplicate'] = df['is_duplicate']

# In[18]:
feature_df.to_csv('./Data/Complete/Features_reduced.csv', sep=',', index=False)
feature_df.to_pickle('./Data/Complete/Features_reduced_pickle', compression='gzip')

# In[17]:
#len(list(feature_df))

