import numpy as np
import pandas as pd
import nltk
import timeit
from nltk.corpus import wordnet
from nltk import word_tokenize


f1 = open('oov_words.csv', 'w')

start_time = timeit.default_timer()
vocab = [w.lower() for w in nltk.corpus.words.words()]
vocab = [w.encode('UTF8') for w in vocab]

train_data = pd.read_csv('Data/train.csv/train.csv')

def modify(ques):
    ques = ' '.join(ques)
    return ques.translate(string.maketrans("",""), string.punctuation)

from nltk.corpus import stopwords
import string
stop = stopwords.words('english')
stop = [x.encode('UTF8') for x in stop]

train_data.fillna('', inplace=True)

frequent = pd.concat([train_data['qid1'] , train_data['qid2']])
freq_qid_vals = frequent.value_counts().head(10)
freq_qid = freq_qid_vals.index.tolist()
qid1_list = train_data['qid1']
qid2_list = train_data['qid2']
qid = pd.concat([qid1_list, qid2_list], ignore_index=True)
questions1_list = train_data['question1']
questions2_list = train_data['question2']
questions_list = pd.concat([questions1_list, questions2_list], ignore_index=True)
questions = pd.concat([qid, questions_list], axis=1)
questions.columns = ['qid', 'question']
questions.drop_duplicates(inplace=True)

questions_list = questions_list.apply(lambda x: [word for word in x.lower().split() if word not in stop])
questions_list = questions_list.apply(lambda x: modify(x))

questions['without_stop'] = questions['question'].apply(lambda x: [ques for ques in x.lower().split() if ques not in stop])
questions['without_stop'] = questions['without_stop'].apply(lambda x: modify(x))

def out_of_vocab(word):
    if word in vocab:
        return False
    elif wordnet.synsets(word.decode('UTF8')):
        return False
    else:
	print>>f1, word
        return True
    
#print [word for word in questions['without_stop'][1000].split() if out_of_vocab(word)]
questions['out_of_vocab'] = questions['without_stop'].apply(lambda ques: [word for word in ques.split() if out_of_vocab(word)])
#print>>f2, questions['out_of_vocab']

elapsed = timeit.default_timer() - start_time
questions.to_csv('oov_words_per_question.csv', index=False)

print elapsed

f1.close()