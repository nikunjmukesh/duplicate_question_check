# coding: utf-8
# In[8]:
import spacy
import pandas as pd
import numpy as np
from tqdm import tqdm


# In[9]:
train_data = pd.read_csv('./Data/train_cleaned.csv')
nlp = spacy.load('en_core_web_lg')


# In[10]:
train_data.fillna('NO QUESTION', inplace=True)


# In[11]:
pos_feature_list = ['lemma', 'POS', 'dependency', 'alpha']
pos_feature_data_1 = pd.DataFrame(columns=pos_feature_list)
pos_feature_data_2 = pd.DataFrame(columns=pos_feature_list)


# In[12]:
def features_1():
    global pos_feature_data_1
    l_list = []
    p_list = []
    d_list = []
    a_list = []
    for i in tqdm(range(len(train_data['question1'])), desc='For Question1 column'):
        lemma_list = []
        pos_list = []
        dep_list = []
        is_alpha_tag = []
        text = nlp(train_data['question1'][i])
        for j in range(len(text)):
            lemma_list.append(text[j].lemma_)
            pos_list.append(text[j].pos_)
            dep_list.append(text[j].dep_)
            if(text[j].is_alpha==True):
                is_alpha_tag.append(1)
            else:
                is_alpha_tag.append(0)
        l_list.append(lemma_list)
        p_list.append(pos_list)
        d_list.append(dep_list)
        a_list.append(is_alpha_tag)
    pos_feature_data_1['lemma'] = l_list
    pos_feature_data_1['POS'] = p_list
    pos_feature_data_1['dependency'] = d_list
    pos_feature_data_1['alpha'] = a_list


# In[13]:
def features_2():
    global pos_feature_data_2
    l_list = []
    p_list = []
    d_list = []
    a_list = []
    for i in tqdm(range(len(train_data['question2'])), desc='For Question2 column'):
        lemma_list = []
        pos_list = []
        dep_list = []
        is_alpha_tag = []
        text = nlp(train_data['question2'][i])
        for j in range(len(text)):
            lemma_list.append(text[j].lemma_)
            pos_list.append(text[j].pos_)
            dep_list.append(text[j].dep_)
            if(text[j].is_alpha==True):
                is_alpha_tag.append(1)
            else:
                is_alpha_tag.append(0)
        l_list.append(lemma_list)
        p_list.append(pos_list)
        d_list.append(dep_list)
        a_list.append(is_alpha_tag)
    pos_feature_data_2['lemma'] = l_list
    pos_feature_data_2['POS'] = p_list
    pos_feature_data_2['dependency'] = d_list
    pos_feature_data_2['alpha'] = a_list


# In[14]:
features_1()


# In[15]:
features_2()


# In[17]:
pos_feature_data_1.to_pickle('nlp_features_1', compression='gzip')
pos_feature_data_2.to_pickle('nlp_features_2', compression='gzip')

