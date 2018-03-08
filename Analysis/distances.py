import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cosine, canberra, euclidean

f1 = open('distances.txt','w')

def create_vectors(ques1, ques2):
    v = CountVectorizer().fit([ques1,ques2])
    v1, v2 = v.transform([ques1,ques2])
    v1 = v1.toarray().ravel()
    v2 = v2.toarray().ravel()
    return [v1,v2]

train_data = pd.read_csv('./Data/train.csv')
train_data.fillna('the', inplace=True)

train_data['cosine_dist'] = " "
train_data['canberra_dist'] = " "
train_data['euclidean_dist'] = " " 
for i in range(len(train_data)):
    print("Calculating for: {}".format(train_data['id'][i]))
    v = CountVectorizer().fit([train_data['question1'][i],train_data['question2'][i]])
    v1, v2 = v.transform([train_data['question1'][i],train_data['question2'][i]])
    v1 = v1.toarray().ravel()
    v2 = v2.toarray().ravel()
    train_data['cosine_dist'][i] = cosine(v1, v2)
    train_data['canberra_dist'][i] = canberra(v1, v2)
    train_data['euclidean_dist'][i] = euclidean(v1, v2)
    print>>f1, (str(train_data['question1'][i]) + "|" + str(train_data['question2'][i]) + "|" + str(train_data['cosine_dist'][i]) + "|" + str(train_data['canberra_dist'][i]) + "|" + str(train_data['euclidean_dist'][i]))
   
#train_data.to_csv('with distances.csv', index=False);
f1.close();
