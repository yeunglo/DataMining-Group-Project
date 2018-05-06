# -*- coding: UTF-8 -*-


import json
import re
import os
import csv

from sklearn.externals import joblib
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.scorer import make_scorer
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import decomposition


t0 = time()
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
a = []
data_list = []
d=[]
tag=[]
tag_super=[]

string=''
string1=''
string2=''
stemmer = SnowballStemmer("english")
with open('C:/Users/admin/Documents/GitHub/DataMining-Group-Project/train_full.json') as Train_data:
   reader1 = json.load(Train_data)
for i in reader1.keys():
    #a.append(i)
    a.append(reader1[i])
for item in a:
    #b = item[0]
    c = item[1]
    # data_list.append(b.keys())
    # string = str(' '.join(b.keys()))
    # string1 = string + ','
    # string2 += string1
    tag.append(c)
    # set the no-toxic,toxic,super-toxic as three classes{0,1,2}
    # if c==1 and item[2]==0:
    #     tag_super.append(1)
    # elif c==1 and item[2]==1:
    #     tag_super.append(2)
    # else:
    #     tag_super.append(0)
words=[]
stems=''
#wordslist=[]
path_full = 'C:/Users/admin/Desktop/dm-group/train.csv'
with open(path_full) as Train:
    reader = csv.DictReader(Train)
    for line in reader:
        words.append((line['comment_text']))
#
# for item in words:
#     # string = str(''.join(i for i in item))
#     # string1 = string + ','
#     # string2 += string1
#     string = str(item)
#     string1 = string + ','
#     string2 += string1

# data_str=string2.split(',')
# data_str = data_str[:-1]
#print data_str
#data_str=[data_str]

tag_str=tag
#print tag_super
#print data_str
#print tag_str
#corpus=[data_str][tag_str]
def getwords(doc):
    splitter = re.compile('\\W+[0-9]*') # split with non-words
    #print(splitter)
    stopworddic = set(stopwords.words('english'))
    words=[s.lower() for s in splitter.split(doc)
           if len(s)>2 and len(s)<20 and s not in stopworddic]
    stems = [stemmer.stem(t) for t in words]
    return dict([(w, stems.count(w)) for w in stems])
    #return stems
def feature_extraction(words,case='tfidf', max_df=0.4, min_df=0):
    files = os.listdir('./')
    saved_file_name = 'doc_matrix_full.pkl'
    if saved_file_name not in files:
        tfidf_vectorizer = TfidfVectorizer(max_df=0.4, max_features=200000, min_df=0.01, tokenizer=getwords,
                                           stop_words='english',
                                           use_idf=True, ngram_range=(1, 1),norm='l2')
        tfidf_matrix = tfidf_vectorizer.fit_transform(words)  # fit the vectorizer to synopses
        joblib.dump(tfidf_matrix, saved_file_name)
    else:
        tfidf_matrix = joblib.load(saved_file_name)
    #self.vec = tfidf_matrix
    #tfidf_matrix=TfidfVectorizer(token_pattern='\w', ngram_range=(1, 2), max_df=max_df, min_df=min_df).fit_transform(data_str)
    return tfidf_matrix
def fitandpredicted(x_train, x_test, tag_train, tag_test, penalty='l2',C=1,solver='lbfgs'):
    #clf = linear_model.LogisticRegressionCV(penalty='l2',class_weight='balanced', solver='lbfgs',multi_class='ovr').fit(x_train, tag_train)
    #clf = linear_model.LogisticRegressionCV(penalty='l2', solver='newton-cg',n_jobs=1,intercept_scaling=1,max_iter=100).fit(x_train, tag_train)
    clf = linear_model.LogisticRegressionCV( ).fit(x_train, tag_train)
    #clf = LogisticRegression(penalty='l2',class_weight='balanced',C=1.0, solver='lbfgs').fit(x_train, tag_train)
    #clf = LogisticRegression(penalty=penalty,C=C, solver=solver, n_jobs=-1).fit(x_train, tag_train)
    predicted = clf.predict(x_test)
    #print(metrics.classification_report(tag_test, predicted))
    print('accuracy_score: %0.5fs' % (metrics.accuracy_score(tag_test, predicted)))


def split_data(input, random_state=15, shuffle=True):
    input_x, y = input
    #print input_x
    x_train, x_test, tag_train, tag_test = train_test_split(input_x, y, test_size=0.3, random_state=random_state)
    return  x_train, x_test, tag_train, tag_test


def train_and_predicted_with_graid(input, param_grid, cv=5):
    print('1')
    tfidf_matrix,tag_str = input
    scoring = ['precision_macro', 'recall_macro', 'f1_macro']
    clf = linear_model.LogisticRegression()
    grid = GridSearchCV(clf, param_grid, cv=cv, scoring='accuracy')
    scores = grid.fit(tfidf_matrix,tag_str)
    print('parameters:')
    best_parameters = grid.best_estimator_.get_params()
    for param_name in sorted(best_parameters):
        print('\t%s: %r' %(param_name, best_parameters[param_name]))
    return scores

print('\t\tuse max_df,min_df=(0.4,0.0) to extract feature,then logistic regression:\t\t')
#print(getwords(words))
#print stems
#max_df = [0.2, 0.4, 0.6, 0.8, 1.0]
#min_df = [0, 0.1, 0.2, 0.3, 0.4]
#stems=getwords(string2)
#print stems[:3]
#print('With min_df=',i)
tfidf_matrix =feature_extraction(words,'tfidf')
#print tfidf_matrix.type
# dist = 1 - cosine_similarity(tfidf_matrix)
# print('2')
# pca = PCA(n_components=10000, whiten=True, random_state=0)
# x = pca.fit_transform(dist)
# n_comp = 500
# svd = decomposition.TruncatedSVD(n_components=n_comp, algorithm='arpack')
# svd.fit(tfidf_matrix)

# mds = MDS(n_components=10000, dissimilarity="precomputed", random_state=1)
# pos = mds.fit_transform(dist)

print(tfidf_matrix.shape)
# n_comp = 100
# svd = decomposition.TruncatedSVD(n_components='n_comp')
tf=tfidf_matrix.toarray()
# x=svd.fit(tf)
pca = PCA(n_components=100, whiten=True, random_state=0)
x=pca.fit_transform(tf)
input=[x,tag_str]
x_train, x_test, tag_train, tag_test = split_data(input)
# print('222')


# # xtrain = pca.fit_transform(x_train)
fitandpredicted(x_train, x_test, tag_train, tag_test)
print('time uesed: %0.4fs' %(time() - t0))


# C= [0.1, 0.2, 0.5, 0.8, 1.5, 3, 5]
# fit_intercept=[True, False]
# penalty=['l2']
# solver=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
# param_grid=dict(C=C, fit_intercept=fit_intercept, penalty=penalty, solver=solver)
# tfidf_matrix = feature_extraction(words, 'tfidf', max_df=1, min_df=0.0)
# input=[tfidf_matrix,tag_str]
#
# scores = train_and_predicted_with_graid(input, cv=5, param_grid=param_grid)


