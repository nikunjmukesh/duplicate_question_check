import Statistical_analysis
import numpy as np
import pandas as pd
import spacy
#import gensim
import cython
from tqdm import tqdm
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


#-----------------LOADING--------------------------------------------------
print('LOADING TRAINING DATA')
#train_data = pd.read_csv("./Data/test_cleaned.csv")
#df = train_data[:100] 
df = pd.read_csv("./Data/test_cleaned.csv")
df.fillna('NO QUESTION', inplace=True)
print('LOADING DONE')

print('LOADING WORDNET')
nlp = spacy.load('en_core_web_lg')
print('WORDNET LOADED')


'''
#-----------------Statistical analysis-------------------------------------
print('PERFORMING STATISTICAL ANALYSIS')
analysis_class = Statistical_analysis.Features(df, "train")
feature_data = analysis_class.features_df()
feature_data.to_pickle("./Data/Feature_data/Statistical_features", compression='gzip')
print('ANALYSIS DONE')
del feature_data

#-----------------Vector creation------------------------------------------
questions = list(df['question1']) + list(df['question2'])
print('PERFORMING TFIDF ANALYSIS')
tfidf = TfidfVectorizer(lowercase=False,)
tfidf.fit_transform(questions)
word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf.idf_))


count = 0
for i in tqdm(questions, desc='CREATING GloVe VECTORS'):
    questions[count] = list(gensim.utils.tokenize(i, deacc=True, lower=True))
    count+=1
glove_model = gensim.models.Word2Vec(questions, size=300, workers=4, iter=10, negative=20)
glove_model.init_sims(replace=True)            #To reduce memory used
print("Number of tokens in glove: ", len(glove_model.wv.vocab))
glove_model.save('./Data/Feature_data/glove_model.mdl')
glove_model.wv.save_word2vec_format('./Data/Feature_data/glove_model.bin', binary=True)
del glove_model
del questions



spacy_vectors = {'question1':list(np.array([doc.vector for doc in tqdm(nlp.pipe(df['question1'], n_threads=50), desc='CREATING SPACY WORD2VEC FOR QUESTION1')])),
                 'question2':list(np.array([doc.vector for doc in tqdm(nlp.pipe(df['question2'], n_threads=50), desc='CREATING SPACY WORD2VEC FOR QUESTION1')]))}
spacy_feature_list = ['q1_f', 'q2_f', 'q1_f_weighted', 'q2_f_weighted']
spacy_feature_df = pd.DataFrame(columns=spacy_feature_list)
spacy_feature_df['q1_f'] = spacy_vectors['question1']
spacy_feature_df['q2_f'] = spacy_vectors['question2']
del spacy_vectors
del spacy_feature_list



q1_vectors = []
q2_vectors = []
for qu in tqdm(list(df['question1']), desc='WEIGHTED VECTORS FOR QUESTION1'):
    doc = nlp(qu)
    mean_vec = np.zeros([len(doc), 300])
    for word in doc:
        vec = word.vector
        try:
            idf = word2tfidf[str(word)]
        except:
            idf = 0
        mean_vec += vec * idf
    mean_vec = mean_vec.mean(axis=0)
    q1_vectors.append(mean_vec)
for qu in tqdm(list(df['question2']), desc='WEIGHTED VECTORS FOR QUESTION2'):
    doc = nlp(qu)
    mean_vec = np.zeros([len(doc), 300])
    for word in doc:
        vec = word.vector
        try:
            idf = word2tfidf[str(word)]
        except:
            idf = 0
        mean_vec += vec * idf
 
    mean_vec = mean_vec.mean(axis=0)
    q2_vectors.append(mean_vec)
spacy_feature_df['q1_f_weighted'] = list(q1_vectors)
spacy_feature_df['q2_f_weighted'] = list(q2_vectors)
spacy_feature_df.to_pickle('./Data/Feature_data/Test_spacy_features_normalized_idf', compression='gzip')
del q1_vectors
del q2_vectors
del spacy_feature_df


print('LOADING GOOGLE NEWS PRE-TRAINED MODEL')
google_vector = gensim.models.KeyedVectors.load_word2vec_format('./Data/GoogleNews-vectors-negative300.bin.gz', binary=True)

model_WS = {'question1':[], 'question2':[]}
for i in tqdm(range(len(df['question1'])), desc='GENERATING WORD VECTORS'):
    model_WS['question1'].append(gensim.models.Word2Vec([df['question1'][i].split()], size=300, window=3, min_count=1, workers=4))
    model_WS['question2'].append(gensim.models.Word2Vec([df['question2'][i].split()], size=300, window=3, min_count=1, workers=4))
#for i in tqdm(range(len(model_WS['question1'])), desc='SAVING WORD VECTORS'):
#    model_WS['question1'][i].wv.save('./Models/question1/{}.csv'.format(i))
#    model_WS['question2'][i].wv.save('./Models/question2/{}.csv'.format(i))
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

df_sen = pd.DataFrame(columns=['question1', 'question2'])
df_sen['question1'] = model_WS_sentences['question1']
df_sen['question2'] = model_WS_sentences['question2']
df_sen.to_pickle('./Data/Feature_data/Test_Sent2vec_non_weighted', compression='gzip')
del model_WS
del model_WS_sentences
del df_sen
'''

#----------------------------NLP FEATURE TAGGING----------------------------------
print('CREATING NLP FEATURES(LEMMA, POS, DEPENDENCY, ALPHAS, NER')
pos_feature_data_1 = pd.DataFrame(columns=['lemma', 'POS', 'dependency', 'alpha'])
pos_feature_data_2 = pd.DataFrame(columns=['lemma', 'POS', 'dependency', 'alpha'])
ner_frame = pd.DataFrame(columns=['question1', 'question2'])

def features_1():
    global pos_feature_data_1
    global ner_frame
    l_list = []
    p_list = []
    d_list = []
    a_list = []
    ner = []
    for i in tqdm(range(len(df['question1'])), desc='For Question1 column'):
        lemma_list = []
        pos_list = []
        dep_list = []
        is_alpha_tag = []
        t_list = []
        text = nlp(df['question1'][i])
        for j in range(len(text)):
            lemma_list.append(text[j].lemma_)
            pos_list.append(text[j].pos_)
            dep_list.append(text[j].dep_)
            if(text[j].is_alpha==True):
                is_alpha_tag.append(1)
            else:
                is_alpha_tag.append(0)
        for j in text.ents:
            t_list.append(j.label_)
        ner.append(t_list)
        l_list.append(lemma_list)
        p_list.append(pos_list)
        d_list.append(dep_list)
        a_list.append(is_alpha_tag)
    #print(str(len(l_list))+','+str(len(pos_feature_data_1['lemma'])))
    pos_feature_data_1['lemma'] = l_list
    pos_feature_data_1['POS'] = p_list
    pos_feature_data_1['dependency'] = d_list
    pos_feature_data_1['alpha'] = a_list
    ner_frame['question1'] = ner

def features_2():
    global pos_feature_data_2
    global ner_frame
    l_list = []
    p_list = []
    d_list = []
    a_list = []
    ner = []
    for i in tqdm(range(len(df['question2'])), desc='For Question2 column'):
        lemma_list = []
        pos_list = []
        dep_list = []
        is_alpha_tag = []
        t_list = []
        text = nlp(df['question2'][i])
        for j in range(len(text)):
            lemma_list.append(text[j].lemma_)
            pos_list.append(text[j].pos_)
            dep_list.append(text[j].dep_)
            if(text[j].is_alpha==True):
                is_alpha_tag.append(1)
            else:
                is_alpha_tag.append(0)
        for j in text.ents:
            t_list.append(j.label_)
        ner.append(t_list)
        l_list.append(lemma_list)
        p_list.append(pos_list)
        d_list.append(dep_list)
        a_list.append(is_alpha_tag)
    pos_feature_data_2['lemma'] = l_list
    pos_feature_data_2['POS'] = p_list
    pos_feature_data_2['dependency'] = d_list
    pos_feature_data_2['alpha'] = a_list
    ner_frame['question2'] = ner

features_1()
features_2()
print("DONE CREATING NLP FEATURES, SAVING FILES")
pos_feature_data_1.to_pickle('./Data/Test/Test_nlp_features_1', compression='gzip')
pos_feature_data_2.to_pickle('./Data/Test/Test_nlp_features_2', compression='gzip')


#-------------------------NLP FEATURE COMPARISON------------------------------------
print("PERFORMING NLP FEATURE COMPARISION")
similar_df = pd.DataFrame(columns=['lemma', 'POS', 'dependency', 'alpha'])
sim_lemma = []
sim_pos = []
sim_dep = []
sim_alpha = []

def lemma_similar():
    for i in tqdm(range(len(pos_feature_data_1['lemma'])), desc='CALCULATING LEMMA SIMILARITIES'):
        if len(pos_feature_data_2['lemma'][i]) >= len(pos_feature_data_1['lemma'][i]):
            c = len(list((Counter(pos_feature_data_1['lemma'][i]) & Counter(pos_feature_data_2['lemma'][i])).elements()))
            sim_lemma.append(c/(len(pos_feature_data_1['lemma'][i])+len(pos_feature_data_2['lemma'][i])))
        else:
            c = len(list((Counter(pos_feature_data_2['lemma'][i]) & Counter(pos_feature_data_1['lemma'][i])).elements()))
            sim_lemma.append(c/(len(pos_feature_data_1['lemma'][i])+len(pos_feature_data_2['lemma'][i])))

def pos_similar():
    for i in tqdm(range(len(pos_feature_data_1['POS'])), desc='CALCULATING POS SIMILARITIES'):
        if len(pos_feature_data_2['POS'][i]) >= len(pos_feature_data_1['POS'][i]):
            c = len(list((Counter(pos_feature_data_1['POS'][i]) & Counter(pos_feature_data_2['POS'][i])).elements()))
            sim_pos.append(c/(len(pos_feature_data_1['POS'][i])+len(pos_feature_data_2['POS'][i])))
        else:
            c = len(list((Counter(pos_feature_data_2['POS'][i]) & Counter(pos_feature_data_1['POS'][i])).elements()))
            sim_pos.append(c/(len(pos_feature_data_1['POS'][i])+len(pos_feature_data_2['POS'][i])))

def dep_similar():
    for i in tqdm(range(len(pos_feature_data_1['dependency'])), desc='CALCULATING DEPENDENCY SIMILARITIES'):
        if len(pos_feature_data_2['dependency'][i]) >= len(pos_feature_data_1['dependency'][i]):
            c = len(list((Counter(pos_feature_data_1['dependency'][i]) & Counter(pos_feature_data_2['dependency'][i])).elements()))
            sim_dep.append(c/(len(pos_feature_data_1['dependency'][i])+len(pos_feature_data_2['dependency'][i])))
        else:
            c = len(list((Counter(pos_feature_data_2['dependency'][i]) & Counter(pos_feature_data_1['dependency'][i])).elements()))
            sim_dep.append(c/(len(pos_feature_data_1['dependency'][i])+len(pos_feature_data_2['dependency'][i])))

def alpha_similar():
    for i in tqdm(range(len(pos_feature_data_1['alpha'])), desc='CALCULATING ALPHABET SIMILARITIES'):
        if(len(set(pos_feature_data_1['alpha'][i]))!=1 | len(set(pos_feature_data_1['alpha'][i]))!=1 ):
            c = sum(1 for j in pos_feature_data_1['alpha'][i] if j==0) + sum(1 for j in pos_feature_data_2['alpha'][i] if j==0)
            sim_alpha.append(c/(len(pos_feature_data_1['alpha'][i]) + len(pos_feature_data_2['alpha'][i])))
        else:
            sim_alpha.append(0) 

lemma_similar()
pos_similar()
dep_similar()
alpha_similar()

similar_df['lemma'] = sim_lemma
similar_df['POS'] = sim_pos
similar_df['dependency'] = sim_dep
similar_df['alpha'] = sim_alpha

similar_df.to_pickle('./Data/Test/Test_NLP_comparison', compression='gzip')


#------------------------NER AND COMPARISON-------------------------------------
dup_list = []
for i in tqdm(range(len(ner_frame['question1'])), desc = 'COMPARING ENTITIES'):
    if (ner_frame['question1'][i] == ner_frame['question2'][i]):
        dup_list.append(len(ner_frame['question1'][i])+0.5)
    else:
        dup_list.append(0)

ner_frame['compare'] = dup_list
ner_frame.to_pickle('./Data/Test/Test_NER_tags', compression='gzip')

  
