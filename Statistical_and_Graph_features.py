
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import networkx as nx
import nltk
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from textstat.textstat import textstat
from collections import defaultdict


# START_INDEX = 0
# END_INDEX = 10000

# In[2]:


feature_list = ['avg_word_length_diff', 'avg_word_length_diff_with_spaces',
                'sentence_length_diff_with_spaces', 'dup_words_diff', 'oov_words_diff', 
                'syllable_count_diff', 'lexicon_count_diff', 'min_kcore', 'max_kcore', 
                'similar_neighbors', 'similar_neighbor_ratio', 'min_freq', 'max_freq']


# In[3]:


train_feature_df = pd.DataFrame(columns=feature_list)
test_feature_df = pd.DataFrame(columns=feature_list)


# train_dataset = pd.read_csv('./Data/train_cleaned.csv')
# train_dataset.fillna('NO QUESTION', inplace=True)
# test_dataset = pd.read_csv('./Data/test_cleaned.csv')
# test_dataset.fillna('NO QUESTION', inplace=True)

# In[4]:


train_df = pd.read_csv('./Data/train_cleaned.csv')
train_df.fillna('NO QUESTION', inplace=True)
test_df = pd.read_csv('./Data/test_cleaned.csv')
test_df.fillna('NO QUESTION', inplace=True)


# train_df = train_dataset[START_INDEX:END_INDEX]
# test_df = test_dataset[START_INDEX:END_INDEX]
# del train_dataset
# del test_dataset

# In[5]:


questions = ""
TYPE_DATA = ""
train_ques1 = train_df['question1']
train_ques2 = train_df['question2']
test_ques1 = test_df['question1']
test_ques2 = test_df['question2']
NUMBER_OF_FEATURES = len(feature_list)


# In[6]:


def avg_word_length_diff(q1, q2):
    avg1 = (len(q1)-q1.count(' '))/(1.0 * len(q1.split()))
    avg2 = (len(q2)-q2.count(' '))/(1.0 * len(q2.split()))
    return avg1-avg2


# In[7]:


def avg_word_length_diff_with_spaces(q1, q2):
    avg1 = len(q1)/(1.0 * len(q1.split()))
    avg2 = len(q2)/(1.0 * len(q2.split()))
    return avg1-avg2


# In[8]:


def sentence_length_diff(q1, q2):
    return len(q1)-len(q2)


# In[9]:


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


# In[10]:


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


# In[11]:


def oov_words_diff(q1, q2):
    return textstat.difficult_words(q1) - textstat.difficult_words(q2)


# In[12]:


def syllable_count_diff(q1, q2):
    return textstat.syllable_count(q1) - textstat.syllable_count(q2)


# In[13]:


def lexicon_count_diff(q1, q2):
    return textstat.lexicon_count(q1) - textstat.lexicon_count(q2)


# In[14]:


def create_hash(train_df, test_df):
    train_q = np.dstack([train_df['question1'], train_df['question2']]).flatten()
    test_q = np.dstack([test_df['question1'], test_df['question2']]).flatten()
    q_complete = np.append(train_q, test_q)
    q_complete = pd.DataFrame(q_complete)[0].drop_duplicates()
    q_complete.reset_index(inplace=True, drop=True)
    q_dict = pd.Series(q_complete.index.values, index=q_complete.values).to_dict()
    return q_dict


# In[15]:


def get_hash(df, dict_hash):
    df['qid1'] = df['question1'].map(dict_hash)
    df['qid2'] = df['question2'].map(dict_hash)
    return df.drop(['question1', 'question2'], axis=1)


# In[25]:


def dict_from_hash(df):
    graph = nx.Graph()
    graph.add_nodes_from(df.qid1)
    edges = list(df[['qid1', 'qid2']].to_records(index=False))
    graph.add_edges_from(edges)
    graph.remove_edges_from(graph.selfloop_edges())
    op_df = pd.DataFrame(data=graph.nodes(), columns=['qid'])
    op_df['kcore'] = 0
    for i in tqdm(range(2, 11), desc='CALCULATING KCORE FEATURES'):                  #2nd argument in range is number of k-core features
        ck = nx.k_core(graph, k=i).nodes()
        print("kcore", i)
        op_df.loc[op_df.qid.isin(ck), 'kcore'] = i
        #print(op_df[:5])
    return op_df.to_dict()['kcore']


# In[17]:


def kcore_from_dict(df, k_dict):
    df['k_q1'] = df['qid1'].apply(lambda x: k_dict[x])
    df['k_q2'] = df['qid2'].apply(lambda x: k_dict[x])
    return df


# In[18]:


def to_minmax(df, col):
    sorted_features = np.sort(np.vstack([df[col + "1"], df[col + "2"]]).T)
    df['min_' + col] = sorted_features[:, 0]
    df['max_' + col] = sorted_features[:, 1]
    return df.drop([col + "1", col + "2"], axis=1)


# In[19]:


def get_neighbors(train_df, test_df):
    neighbors = defaultdict(set)
    for i in [train_df, test_df]:
        for q1, q2 in zip(i['qid1'], i['qid2']):
            neighbors[q1].add(q2)
            neighbors[q2].add(q1)
    return neighbors


# In[20]:


def neighbor_similarity(df, neighbors):
    common = df.apply(lambda x: len(neighbors[x.qid1].intersection(neighbors[x.qid2])), axis=1)
    min_n = df.apply(lambda x: min(len(neighbors[x.qid1]), len(neighbors[x.qid2])), axis=1)
    df['similar_neighbor_ratio'] = common/min_n
    df['similar_neighbors'] = common.apply(lambda x: min(x, 5)) #2nd argument is upper bound for number of neighbors
    return df


# In[21]:


def freq_features(df, freq_map):
    df['freq1'] = df['qid1'].map(lambda x: min(freq_map[x], 100))  #2nd argument in min is upper bound for frequencies
    df['freq2'] = df['qid2'].map(lambda x: min(freq_map[x], 100))
    return df


# FEATURE GENERATION DRIVER CODE

# In[ ]:


print("-------------CALCULATING FEATURES------------------------------------------------------")


# In[ ]:


print("-------------STATISTICAL FEATURES------------------------------------------------------")


# In[ ]:


for i in tqdm(range(START_INDEX, END_INDEX), desc='CREATING STATISTICAL FEATURES FOR TRAIN'):
    feature = np.empty(NUMBER_OF_FEATURES) * np.nan
    feature[0] = avg_word_length_diff(train_ques1[i], train_ques2[i])
    feature[1] = avg_word_length_diff_with_spaces(train_ques1[i], train_ques2[i])
    feature[2] = sentence_length_diff(train_ques1[i], train_ques2[i])
    feature[3] = dup_words_diff(train_ques1[i], train_ques2[i])
    feature[4] = oov_words_diff(train_ques1[i], train_ques2[i])
    feature[5] = syllable_count_diff(train_ques1[i], train_ques2[i])
    feature[6] = lexicon_count_diff(train_ques1[i], train_ques2[i])
    train_feature_df.loc[len(train_feature_df)] = feature


# In[ ]:


for i in tqdm(range(START_INDEX, END_INDEX), desc='CREATING STATISTICAL FEATURES FOR TEST'):
    feature = np.empty(NUMBER_OF_FEATURES) * np.nan
    feature[0] = avg_word_length_diff(test_ques1[i], test_ques2[i])
    feature[1] = avg_word_length_diff_with_spaces(test_ques1[i], test_ques2[i])
    feature[2] = sentence_length_diff(test_ques1[i], test_ques2[i])
    feature[3] = dup_words_diff(test_ques1[i], test_ques2[i])
    feature[4] = oov_words_diff(test_ques1[i], test_ques2[i])
    feature[5] = syllable_count_diff(test_ques1[i], test_ques2[i])
    feature[6] = lexicon_count_diff(test_ques1[i], test_ques2[i])
    test_feature_df.loc[len(test_feature_df)] = feature


# In[ ]:


print("--------------GRAPH FEATURES-----------------------------------------------------------")


# In[22]:


q_dict = create_hash(train_df, test_df)
train_df = get_hash(train_df, q_dict)
test_df = get_hash(test_df, q_dict)
print(">>>>>>>TOTAL NUMBER OF UNIQUE QUESTIONS = ", len(q_dict))


# In[23]:


df_total = pd.concat([train_df, test_df])


# In[26]:


k_dict = dict_from_hash(df_total)


# In[27]:


train_df = kcore_from_dict(train_df, k_dict)
test_df = kcore_from_dict(test_df, k_dict)


# In[29]:


train_df = to_minmax(train_df, "k_q")
test_df = to_minmax(test_df, "k_q")


# In[30]:


neighbors = get_neighbors(train_df, test_df)
train_df = neighbor_similarity(train_df, neighbors)
test_df = neighbor_similarity(test_df, neighbors)


# In[31]:


f_map = dict(zip(*np.unique(np.vstack((df_total['qid1'], df_total['qid2'])), return_counts=True)))
train_df = freq_features(train_df, f_map)
test_df = freq_features(test_df, f_map)


# In[32]:


train_df = to_minmax(train_df, 'freq')
test_df = to_minmax(test_df, 'freq')


# In[ ]:


test_feature_df['min_kcore'] = test_df['min_k_q']
test_feature_df['max_kcore'] = test_df['max_k_q']
test_feature_df['similar_neighbors'] = test_df['similar_neighbors']
test_feature_df['similar_neighbor_ratio'] = test_df['similar_neighbor_ratio']
test_feature_df['min_freq'] = test_df['min_freq']
test_feature_df['max_freq'] = test_df['max_freq']
del test_df
test_feature_df.to_pickle('./Features/Train/Stat_Graph_features', compression='gzip')
del test_feature_df


# In[ ]:


train_feature_df['min_kcore'] = train_df['min_k_q']
train_feature_df['max_kcore'] = train_df['max_k_q']
train_feature_df['similar_neighbors'] = train_df['similar_neighbors']
train_feature_df['similar_neighbor_ratio'] = train_df['similar_neighbor_ratio']
train_feature_df['min_freq'] = train_df['min_freq']
train_feature_df['max_freq'] = train_df['max_freq']
del train_df
train_feature_df.to_pickle('./Features/Train/Stat_Graph_features', compression='gzip')

