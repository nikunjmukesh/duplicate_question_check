# coding: utf-8
# In[1]:
import spacy
import numpy as np
import pandas as pd
import gensim
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:
nlp = spacy.load('en_core_web_lg')


# In[3]:
df = pd.read_csv('./Data/Cleaned_train_lac.csv')
df.fillna('NO QUESTION', inplace = True)


# In[4]:
questions = list(df['question1']) + list(df['question2'])


# In[5]:
tfidf = TfidfVectorizer(lowercase=False,)


# In[6]:
tfidf.fit_transform(questions)


# In[7]:
word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf.idf_))


# CREATING GLOVE VECTORS HERE
# In[8]:


# In[9]:


count = 0
for i in tqdm(questions):
    questions[count] = list(gensim.utils.tokenize(i, deacc=True, lower=True))
    count+=1
glove_model = gensim.models.Word2Vec(questions, size=300, workers=4, iter=10, negative=20)


# In[10]:


glove_model.init_sims(replace=True)            #To reduce memory used
print("Number of tokens in glove: ", len(glove_model.wv.vocab))


# In[11]:


glove_model.save('./Models/glove_model.mdl')
glove_model.wv.save_word2vec_format('./Models/glove_model.bin', binary=True)


# CREATING SPACY WORD VECTORS USING WIKIPEDIA TRAINED MODEL

# In[14]:


spacy_vectors = {'question1':list(np.array([doc.vector for doc in tqdm(nlp.pipe(df['question1'], n_threads=25))])),
                 'question2':list(np.array([doc.vector for doc in tqdm(nlp.pipe(df['question2'], n_threads=25))]))}
#q1_vec = [doc.vector for doc in nlp.pipe(df['question1'], n_threads=25)]
#q2_vec = [doc.vector for doc in nlp.pipe(df['question2'], n_threads=25)]
#q1_vec = list(np.array([doc.vector for doc in nlp.pipe(df['question1'], n_threads=25)]))
#q2_vec = list(np.array([doc.vector for doc in nlp.pipe(df['question2'], n_threads=25)]))


# In[16]:


spacy_feature_df = ['q1_f', 'q2_f', 'q1_f_weighted', 'q2_f_weighted']
feature_df = pd.DataFrame(columns=spacy_feature_df)
feature_df['q1_f'] = spacy_vectors['question1']
feature_df['q2_f'] = spacy_vectors['question2']


# LOAD WORD WISE WORD2VEC MODELS FROM DISK NOW AND THEN CREATE WEIGHTED WORD2VEC SENTENCE VECTORS FOR EACH QUESTION
# PERFORM SAME TASK WITH SPACY WORD2VEC VECTORS AS WELL

# In[27]:


#creating weighted vectors for question1 column
q1_vectors = []
for qu in tqdm(list(df['question1']), desc='WEIGHTED VECTORS FOR QUESTION1'):
    doc = nlp(qu)
    mean_vec = np.zeros([len(doc), 300])
    for word in doc:
        vec = word.vector
        try:
            idf = word2tfidf[str(word)]
        except:
            idf = 1
        mean_vec += vec * idf
    mean_vec = mean_vec.mean(axis=0)
    q1_vectors.append(mean_vec)
feature_df['q1_f_weighted'] = list(q1_vectors)


# In[28]:


#creating weighted vectors for question1 column
q2_vectors = []
for qu in tqdm(list(df['question2'])):
    doc = nlp(qu)
    mean_vec = np.zeros([len(doc), 300])
    for word in doc:
        vec = word.vector
        try:
            idf = word2tfidf[str(word)]
        except:
            idf = 1
        mean_vec += vec * idf
 
    mean_vec = mean_vec.mean(axis=0)
    q2_vectors.append(mean_vec)
feature_df['q2_f_weighted'] = list(q2_vectors)


# In[29]:


#UNCOMMENT THE NEXT LINE TO SAVE ALL DATA TO A SINGLE FILE
feature_df.to_pickle('./Data/spacy_features_non_normalized_idf.csv', compression='gzip')

