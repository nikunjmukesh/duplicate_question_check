import numpy as np
import pandas as pd
import nltk
from tqdm import tqdm
from collections import Counter
from textstat.textstat import textstat

#START_INDEX = 0
#END_INDEX = 100


feature_list = ['avg_word_length_diff', 'avg_word_length_diff_with_spaces',
                'sentence_length_diff_with_spaces', 'dup_words_diff', 'oov_words_diff', 
                'syllable_count_diff', 'lexicon_count_diff']

train_feature_df = pd.DataFrame(columns=feature_list)
#test_feature_df = pd.DataFrame(columns=feature_list)

#train_dataset = pd.read_csv('./Data/train_cleaned.csv')
#train_dataset.fillna('NO QUESTION', inplace=True)
#test_dataset = pd.read_csv('./Data/test_cleaned.csv')
#test_dataset.fillna('NO QUESTION', inplace=True)

train_df = pd.read_csv('./Data/train_cleaned.csv')
train_df.fillna('NO QUESTION', inplace=True)
#test_df = pd.read_csv('./Data/test_cleaned.csv')
#test_df.fillna('NO QUESTION', inplace=True)

#train_df = train_dataset[START_INDEX:END_INDEX]
#test_df = test_dataset[START_INDEX:END_INDEX]
#del train_dataset
#del test_dataset


questions = ""
train_ques1 = train_df['question1']
train_ques2 = train_df['question2']
#test_ques1 = test_df['question1']
#test_ques2 = test_df['question2']
NUMBER_OF_FEATURES = len(feature_list)

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


def oov_words_diff(q1, q2):
    return textstat.difficult_words(q1) - textstat.difficult_words(q2)


def syllable_count_diff(q1, q2):
    return textstat.syllable_count(q1) - textstat.syllable_count(q2)


def lexicon_count_diff(q1, q2):
    return textstat.lexicon_count(q1) - textstat.lexicon_count(q2)

for i in tqdm(range(len(train_df['question1'])), desc='CREATING STATISTICAL FEATURES FOR TRAIN'):
    feature = np.empty(NUMBER_OF_FEATURES) * np.nan
    feature[0] = avg_word_length_diff(train_ques1[i], train_ques2[i])
    feature[1] = avg_word_length_diff_with_spaces(train_ques1[i], train_ques2[i])
    feature[2] = sentence_length_diff(train_ques1[i], train_ques2[i])
    feature[3] = dup_words_diff(train_ques1[i], train_ques2[i])
    feature[4] = oov_words_diff(train_ques1[i], train_ques2[i])
    feature[5] = syllable_count_diff(train_ques1[i], train_ques2[i])
    feature[6] = lexicon_count_diff(train_ques1[i], train_ques2[i])
    train_feature_df.loc[len(train_feature_df)] = feature

print('Saving pickle')
train_feature_df.to_pickle('./Features/Train/Statistical_features', compression='gzip')
print('Saving csv')
train_feature_df.to_csv('./Features/Train/Statistical_features.csv', index=False)