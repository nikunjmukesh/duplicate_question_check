# coding: utf-8
# In[1]:
import spacy
import numpy as np
import pandas as pd
from tqdm import tqdm

# In[2]:
nlp = spacy.load('en_core_web_lg')


# In[4]:
df = pd.read_csv('./Data/test_cleaned.csv')


# In[5]:
df.fillna('NO QUESTION', inplace=True)


# In[6]:
ner_frame = pd.DataFrame(columns=['question1', 'question2'])


# In[7]:
ner_1 = []
ner_2 = []
for i in tqdm(range(len(df['question1'])), desc='TAGGING NAMED ENTITIES'):
    t_list_1 = []
    t_list_2 = []
    text_1 = nlp(df['question1'][i])
    text_2 = nlp(df['question2'][i])
    for j in text_1.ents:
        t_list_1.append(j.label_)
    for j in text_2.ents:
        t_list_2.append(j.label_)
    ner_1.append(t_list_1)
    ner_2.append(t_list_2)


# In[52]:
ner_frame['question1'] = ner_1
ner_frame['question2'] = ner_2


# In[59]:
dup_list = []
for i in tqdm(range(len(ner_frame['question1'])), desc = 'COMPARING ENTITIES'):
    if (ner_frame['question1'][i] == ner_frame['question2'][i]):
        dup_list.append(len(ner_frame['question1'][i])+0.5)
    else:
        dup_list.append(0)      


# In[60]:
ner_frame['compare'] = dup_list


# In[62]:
ner_frame.to_pickle('./Data/Test_NER_tags', compression='gzip')

