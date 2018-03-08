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


class Features:
    #df_feature = pd.DataFrame(columns=['avg_word_length_diff', 'avg_word_length_diff_with_spaces', 'sentence_length_diff_with_spaces', 'cosine', 'canberra', 'euclidean', 'braycurtis', 'chebyshev', 'cityblock', 'correlation', 'dice', 'hamming', 'jaccard', 'russellrao', 'sokalmichener', 'sokalsneath', 'sum_of_word_prob_diff', 'dup_words_diff', 'oov_words_diff', 'syllable_count_diff', 'lexicon_count_diff'])
    df_feature = pd.DataFrame(columns=['avg_word_length_diff', 'avg_word_length_diff_with_spaces', 'sentence_length_diff_with_spaces', 'cosine', 'canberra', 'euclidean', 'braycurtis', 'chebyshev', 'cityblock', 'correlation', 'dice', 'hamming', 'jaccard', 'russellrao', 'dup_words_diff', 'oov_words_diff', 'syllable_count_diff', 'lexicon_count_diff'])
    questions = ""
    fdist = nltk.FreqDist()
    type_data = ""
    #vocab = ""
    n = 0
    def __init__(self, questions, type_data):
        self.ques1 = questions['question1']
        self.ques2 = questions['question2']
        self.type_data = type_data
        if (type_data == "train"):
            self.is_duplicate = questions['is_duplicate']
            #self.df_feature = pd.DataFrame(columns=['avg_word_length_diff', 'avg_word_length_diff_with_spaces', 'sentence_length_diff_with_spaces', 'cosine', 'canberra', 'euclidean', 'braycurtis', 'chebyshev', 'cityblock', 'correlation', 'dice', 'hamming', 'jaccard', 'russellrao', 'sokalmichener', 'sokalsneath', 'sum_of_word_prob_diff', 'dup_words_diff', 'oov_words_diff', 'syllable_count_diff', 'lexicon_count_diff', 'is_duplicate'])
            self.df_feature = pd.DataFrame(columns=['avg_word_length_diff', 'avg_word_length_diff_with_spaces', 'sentence_length_diff_with_spaces', 'cosine', 'canberra', 'euclidean', 'braycurtis', 'chebyshev', 'cityblock', 'correlation', 'dice', 'hamming', 'jaccard', 'russellrao', 'dup_words_diff', 'oov_words_diff', 'syllable_count_diff', 'lexicon_count_diff', 'is_duplicate'])
        #self.build_vocab()
        #self.calculate_fdist_all_questions()
        self.n = len(self.df_feature.columns)
        
    def avg_word_length_diff(self, q1, q2):
        #print q1
        avg1 = (len(q1)-q1.count(' '))/(1.0 * len(q1.split()))
        avg2 = (len(q2)-q2.count(' '))/(1.0 * len(q2.split()))
        return avg1-avg2
    
    def avg_word_length_diff_with_spaces(self, q1, q2):
        #print q1
        avg1 = len(q1)/(1.0 * len(q1.split()))
        avg2 = len(q2)/(1.0 * len(q2.split()))
        return avg1-avg2
    
    def sentence_length_diff(self, q1, q2):
        return len(q1)-len(q2)
    
    def dist_preproc(self, q1, q2):
        v = CountVectorizer().fit([q1, q2])
        vec1, vec2 = v.transform([q1, q2])
        v1 = vec1.toarray().ravel()
        v2 = vec2.toarray().ravel()
        return v1, v2
    
    def cosine_distance(self, v1, v2):
        return cosine(v1, v2)
    
    def canberra_distance(self, v1, v2):
        return canberra(v1, v2)
    
    def euclidean_distance(self, v1, v2):
        return euclidean(v1, v2) 
    
    def braycurtis_distance(self, v1, v2):
        return braycurtis(v1, v2)
    
    def chebyshev_distance(self, v1, v2):
        return chebyshev(v1, v2)
    
    def cityblock_distance(self, v1, v2):
        return cityblock(v1, v2)
    
    def correlation_distance(self, v1, v2):
        return correlation(v1, v2)
    
    def dice_distance(self, v1, v2):
        return dice(v1, v2)
    
    def hamming_distance(self, v1, v2):
        return hamming(v1, v2)
    
    def jaccard_distance(self, v1, v2):
        return jaccard(v1, v2)
    
    def russellrao_distance(self, v1, v2):
        return russellrao(v1, v2)
    
    #def sokalmichener_distance(self, v1, v2):
    #    return sokalmichener(v1, v2)
    
    def sokalsneath_distance(self, v1, v2):
        return sokalsneath(v1, v2)
    
    def distances(self, q1, q2):
        v1, v2 = self.dist_preproc(q1, q2)
        return self.cosine_distance(v1, v2), self.canberra_distance(v1, v2), self.euclidean_distance(v1, v2), self.braycurtis_distance(v1, v2), self.chebyshev_distance(v1, v2), self.cityblock_distance(v1, v2), self.correlation_distance(v1, v2), self.dice_distance(v1, v2), self.hamming_distance(v1, v2), self.jaccard_distance(v1, v2),  self.russellrao_distance(v1, v2) #self.sokalsneath_distance(v1, v2) #self.sokalmichener_distance(v1, v2), 
    
    def num_duplicate_words_diff(self, q1, q2):
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
    def dup_words_diff(self, q1, q2):
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
    
    def build_vocab(self):
        #self.vocab = [w.lower() for w in nltk.corpus.words.words()]
        #self.vocab = [w.encode('UTF8') for w in self.vocab]
        #print self.vocab
        f1 = open("./Data/snli_1.0_train.txt", 'r')
        self.vocab = textstat.difficult
        f1.close()
        
    def oov_words_diff(self, q1, q2):
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
    
    def syllable_count_diff(self, q1, q2):
        return textstat.syllable_count(q1) - textstat.syllable_count(q2)
    
    def lexicon_count_diff(self, q1, q2):
        return textstat.lexicon_count(q1) - textstat.lexicon_count(q2)
                 
    def features_df(self):
        for i in tqdm(range(len(self.ques1)), desc='CREATING STATISTICAL FEATURES'):
            if (self.type_data == "train"):
                feature = np.empty(self.n) * np.nan
            else:
                feature = np.empty(self.n-1) * np.nan
            feature[0] = self.avg_word_length_diff(self.ques1[i], self.ques2[i])
            feature[1] = self.avg_word_length_diff_with_spaces(self.ques1[i], self.ques2[i])
            feature[2] = self.sentence_length_diff(self.ques1[i], self.ques2[i])
            feature[3], feature[4], feature[5], feature[6], feature[7], feature[8], feature[9], feature[10], feature[11], feature[12], feature[13] = self.distances(self.ques1[i], self.ques2[i])
            #feature[16] = self.sum_of_word_prob_diff(self.ques1[i], self.ques2[i])
            feature[14] = self.dup_words_diff(self.ques1[i], self.ques2[i])
            feature[15] = self.oov_words_diff(self.ques1[i], self.ques2[i])
            feature[16] = self.syllable_count_diff(self.ques1[i], self.ques2[i])
            feature[17] = self.lexicon_count_diff(self.ques1[i], self.ques2[i])
            if (self.type_data == "train"):
                feature[self.n-1] = self.is_duplicate[i]
            self.df_feature.loc[len(self.df_feature)] = feature
            #self.count_total_words()
        return self.df_feature

'''train_small = pd.read_csv("Cleaned train data small.csv")
train_small = train_small[:50] 
train_small.fillna("NULL")
analysis_class = Analysis(train_small, "train")
feature_data = analysis_class.analysis_df()
feature_data.to_csv("features.csv")'''




