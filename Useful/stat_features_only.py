#import Statistical_analysis
import numpy as np
import pandas as pd
import nltk
#import spacy
#import gensim
#import cython
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cosine, canberra, euclidean, braycurtis, chebyshev, cityblock, correlation, dice, hamming, jaccard, russellrao, sokalmichener, sokalsneath
from collections import Counter
from textstat.textstat import textstat

#CHANGE THESE VALUES FOR YOUR SLICE OF THE TEST FILE
START_INDEX = 1400000
END_INDEX = 1500000

#NIKUNJ CHANGE START_INDEX TO 1500000 END_INDEX TO 2100000
#NIHARIKA CHANGE START_INDEX TO 2100000 CHANGE SLICE END TO NOTHING

#-----------------LOADING--------------------------------------------------
print('LOADING TRAINING DATA')
train_data = pd.read_csv("./Data/test_cleaned.csv")
df = train_data[START_INDEX:END_INDEX] 
#df = pd.read_csv("./Data/test_cleaned.csv")
df.fillna('NO QUESTION', inplace=True)
print('LOADING DONE '+str(START_INDEX))
del train_data


#-----------------STAT CODE------------------------------------------------
print('PERFORMING STATISTICAL ANALYSIS')
feature_df = pd.DataFrame(columns=['avg_word_length_diff', 'avg_word_length_diff_with_spaces', 'sentence_length_diff_with_spaces', 'cosine', 'canberra', 'euclidean', 'braycurtis', 'chebyshev', 'cityblock', 'correlation', 'dice', 'hamming', 'jaccard', 'russellrao', 'dup_words_diff', 'oov_words_diff', 'syllable_count_diff', 'lexicon_count_diff'])
questions = ""
#fdist = nltk.FreqDist()
type_data = ""
#vocab = ""
n = 0
ques1 = df['question1']
ques2 = df['question2']
n = len(feature_df.columns)
type_data='test'

def avg_word_length_diff(q1, q2):
    avg1 = (len(q1)-q1.count(' '))/(1.0 * len(q1.split()))
    avg2 = (len(q2)-q2.count(' '))/(1.0 * len(q2.split()))
    return avg1-avg2
    
def avg_word_length_diff_with_spaces(q1, q2):
    avg1 = len(q1)/(1.0 * len(q1.split()))
    avg2 = len(q2)/(1.0 * len(q2.split()))
    return avg1-avg2
    
def sentence_length_diff(q1, q2):
    return len(q1)-len(q2)
    
def dist_preproc(q1, q2):
    v = CountVectorizer().fit([q1, q2])
    vec1, vec2 = v.transform([q1, q2])
    v1 = vec1.toarray().ravel()
    v2 = vec2.toarray().ravel()
    return v1, v2
    
def cosine_distance(v1, v2):
    return cosine(v1, v2)
    
def canberra_distance(v1, v2):
    return canberra(v1, v2)
    
def euclidean_distance(v1, v2):
    return euclidean(v1, v2) 
    
def braycurtis_distance(v1, v2):
    return braycurtis(v1, v2)
    
def chebyshev_distance(v1, v2):
    return chebyshev(v1, v2)
    
def cityblock_distance(v1, v2):
    return cityblock(v1, v2)
    
def correlation_distance(v1, v2):
    return correlation(v1, v2)
    
def dice_distance(v1, v2):
    return dice(v1, v2)
    
def hamming_distance(v1, v2):
    return hamming(v1, v2)
    
def jaccard_distance(v1, v2):
    return jaccard(v1, v2)
    
def russellrao_distance(v1, v2):
    return russellrao(v1, v2)
    
#def sokalmichener_distance(self, v1, v2):
#    return sokalmichener(v1, v2)
    
def sokalsneath_distance(v1, v2):
    return sokalsneath(v1, v2)
    
def distances(q1, q2):
    v1, v2 = dist_preproc(q1, q2)
    return cosine_distance(v1, v2), canberra_distance(v1, v2), euclidean_distance(v1, v2), braycurtis_distance(v1, v2), chebyshev_distance(v1, v2), cityblock_distance(v1, v2), correlation_distance(v1, v2), dice_distance(v1, v2), hamming_distance(v1, v2), jaccard_distance(v1, v2),  russellrao_distance(v1, v2) #self.sokalsneath_distance(v1, v2) #self.sokalmichener_distance(v1, v2), 
    
def num_duplicate_words_diff(q1, q2):
    counts1 = Counter(q1.split())
    counts2 = Counter(q2.split())
    sum1 = sum2 = 0
    for word, count in counts1.items():
        if count>1:
            sum1 += sum1
    for word, count in counts2.items():
        if count>1:
            sum2 += sum2
    return sum1-sum2
    
'''
def calculate_fdist_all_questions(self):
    for i in tqdm(range(len(self.ques1)), desc='CALCULATING FREQDIST FOR ALL'):
        self.questions = self.questions + " " + self.ques1[i] + " " + self.ques2[i]
    self.fdist = nltk.FreqDist(self.questions.split())
    

def sum_of_word_prob_diff(self, q1, q2):
    words1 = [word for word in q1.split()]
    words2 = [word for word in q2.split()]
    sum1 = sum2 = 0.0
    for i in words1:
        sum1 += self.fdist.freq(i)
    for j in words2:
        sum2 += self.fdist.freq(j)
    return sum1-sum2
'''
def dup_words_diff(q1, q2):
    words1 = q1.split()
    words2 = q2.split()
    fd1 = nltk.FreqDist(words1)
    fd2 = nltk.FreqDist(words2)
    nd1 = nd2 = 0.0
    for w, f in fd1.items():
        if f>1:
            nd1 = nd1 + 1
    nd1 = nd1/(1.0*len(words1))
    for w, f in fd2.items():
        if f>1:
            nd2 = nd2 + 1
    nd2 = nd2/(1.0*len(words2))
    return nd1-nd2

'''    
def build_vocab(self):
    #self.vocab = [w.lower() for w in nltk.corpus.words.words()]
    #self.vocab = [w.encode('UTF8') for w in self.vocab]
    #print self.vocab
    f1 = open("./Data/snli_1.0_train.txt", 'r')
    self.vocab = textstat.difficult
    f1.close()
 '''     
 
def oov_words_diff(q1, q2):
    #words1 = q1.split()
    #words2 = q2.split()
    #oov1 = oov2 = 0
    return textstat.difficult_words(q1) - textstat.difficult_words(q2)
    #for w in words1:
    #    if w not in self.vocab:
    #        #print "Q1: " + w
    #        oov1 = oov1 + 1
    #for w in words2:
    #    if w not in self.vocab:
    #        #print "Q2: " + w
    #        oov2 = oov2 + 1
    #return oov1 - oov2
    #return textstat.difficult_words(q1) - textstat.difficult_words(q2)
    
def syllable_count_diff(q1, q2):
    return textstat.syllable_count(q1) - textstat.syllable_count(q2)
    
def lexicon_count_diff(q1, q2):
    return textstat.lexicon_count(q1) - textstat.lexicon_count(q2)
                 
for i in tqdm(range(START_INDEX, END_INDEX), desc='CREATING STATISTICAL FEATURES'):
    if (type_data == "train"):
        feature = np.empty(n) * np.nan
    else:
        feature = np.empty(n) * np.nan
    feature[0] = avg_word_length_diff(ques1[i], ques2[i])
    feature[1] = avg_word_length_diff_with_spaces(ques1[i], ques2[i])
    feature[2] = sentence_length_diff(ques1[i], ques2[i])
    feature[3], feature[4], feature[5], feature[6], feature[7], feature[8], feature[9], feature[10], feature[11], feature[12], feature[13] = distances(ques1[i], ques2[i])
    #feature[16] = self.sum_of_word_prob_diff(self.ques1[i], self.ques2[i])
    feature[14] = dup_words_diff(ques1[i], ques2[i])
    feature[15] = oov_words_diff(ques1[i], ques2[i])
    feature[16] = syllable_count_diff(ques1[i], ques2[i])
    feature[17] = lexicon_count_diff(ques1[i], ques2[i])
    if (type_data == "train"):
        feature[n-1] = is_duplicate[i]
    feature_df.loc[len(feature_df)] = feature
    #self.count_total_words()


    #feature_list = ['avg_word_length_diff', 'avg_word_length_diff_with_spaces',
    #            'sentence_length_diff_with_spaces', 'dup_words_diff', 'oov_words_diff', 
    #            'syllable_count_diff', 'lexicon_count_diff', 'min_kcore', 'max_kcore', 
    #            'similar_neighbors', 'similar_neighbor_ratio', 'min_freq', 'max_freq']
    

'''

#-----------------Statistical analysis-------------------------------------

analysis_class = Statistical_analysis.Features(df, "train")
feature_data = analysis_class.features_df()
'''
#feature_df.to_csv("./Data/Feature_data/Statistical_features.csv")
feature_df.to_pickle("./Data/Feature_data/Statistical_features_{}".format(int(START_INDEX)), compression='gzip')
print('ANALYSIS DONE')
del feature_df
