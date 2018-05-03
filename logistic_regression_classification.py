# -*- coding: UTF-8 -*-

import json

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.scorer import make_scorer
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from time import time
from sklearn.model_selection import GridSearchCV

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
a = []
data_list = []
d=[]
tag=[]
tag_super=[]
t0 = time()
string=''
string1=''
string2=''
with open('C:/Users/admin/Documents/GitHub/DataMining-Group-Project/train_full.json') as Train_data:
   reader1 = json.load(Train_data)
#reader1={"78915629f8418c85": [{"and": 1, "hair": 1, "shaggy": 1, "also": 1, "very": 1, "some": 1,"dark": 1,"jewish": 1,"looks": 1,"imagine": 1,"you": 1,"beard": 1,"with": 1,"him": 1,"can": 1}, 1, 0, 0, 0, 0, 0], "ae6308347d82cb41": [{"about": 1, "think": 1, "your": 1, "hgbob": 1, "edit": 1, "please": 1, "some": 1, "here": 1, "you": 1, "may": 1, "see": 1, "flags": 1, "controversial": 1, "ask": 1, "the": 1, "debate": 1, "talk": 1, "theres": 1}, 1, 0, 0, 0, 0, 0], "bdf44cf93bd918a3": [{"redirect": 1, "despot": 1, "talkjohn": 1, "palaiologos": 1}, 0, 0, 0, 0, 0, 0], "291708466f6a09e4": [{"all": 1, "jimim": 1, "lout": 1}, 1, 1, 0, 0, 0, 0]}
#reader1={"78915629f8418c85": [{"and": 1, "hair": 1, "shaggy": 1, "also": 1, "very": 1, "some": 1, "dark": 1, "jewish": 1, "looks": 1, "imagine": 1, "you": 1, "beard": 1, "with": 1, "him": 1, "can": 1}, 0, 0, 0, 0, 0, 0], "797850ad54572dc0": [{"and": 1, "force": 1, "centrifugal": 1, "effect": 1, "elucidating": 1, "way": 1, "radial": 1, "the": 1, "outward": 1, "best": 1}, 0, 0, 0, 0, 0, 0], "cac87a20749a22be": [{"think": 1, "wpblar": 1, "her": 1, "standalone": 1, "marginal": 1, "should": 1, "lpradicals": 1, "article": 1, "implemented": 1, "notability": 2, "best": 1}, 0, 0, 0, 0, 0, 0], "ae6308347d82cb41": [{"about": 1, "think": 1, "your": 1, "hgbob": 1, "edit": 1, "please": 1, "some": 1, "here": 1, "you": 1, "may": 1, "see": 1, "flags": 1, "controversial": 1, "ask": 1, "the": 1, "debate": 1, "talk": 1, "theres": 1}, 0, 0, 0, 0, 0, 0], "bdf44cf93bd918a3": [{"redirect": 1, "despot": 1, "talkjohn": 1, "palaiologos": 1}, 0, 0, 0, 0, 0, 0], "291708466f6a09e4": [{"all": 1, "jimim": 1, "lout": 1}, 0, 0, 0, 0, 0, 0]}
#json_str = json.dumps(reader1)
#print json_str
for i in reader1.keys():
    #a.append(i)
    a.append(reader1[i])

for item in a:
    b = item[0]
    c = item[1]
    data_list.append(b.keys())
    string = str(' '.join(b.keys()))
    string1 = string + ','
    string2 += string1
    tag.append(c)
    if c==1 and item[2]==0:
        tag_super.append(1)
    elif c==1 and item[2]==1:
        tag_super.append(2)
    else:
        tag_super.append(0)


data_str=string2.split(',')
data_str = data_str[:-1]
#data_str=[data_str]
tag_str=tag
print tag_super
#print data_str
print tag_str
#corpus=[data_str][tag_str]
def feature_extraction(data_str,case='tfidf', max_df=1.0, min_df=0.0):
    tfidf_matrix=TfidfVectorizer(token_pattern='\w', ngram_range=(1, 2), max_df=max_df, min_df=min_df).fit_transform(data_str)
    return tfidf_matrix

def fitandpredicted(x_train, x_test, tag_train, tag_test, penalty='l2',C=1,solver='lbfgs'):
    clf = linear_model.LogisticRegressionCV(penalty='l2',class_weight='balanced', solver='lbfgs').fit(x_train, tag_train)
    #clf = LogisticRegression(penalty='l2',class_weight='balanced',C=1.0, solver='lbfgs').fit(x_train, tag_train)
    #clf = LogisticRegression(penalty=penalty,C=C, solver=solver, n_jobs=-1).fit(x_train, tag_train)
    predicted = clf.predict(x_test)
    #print(metrics.classification_report(tag_test, predicted))
    print('accuracy_score: %0.5fs' % (metrics.accuracy_score(tag_test, predicted)))


def split_data(input, random_state=10, shuffle=True):
    input_x, y = input
    #print input_x
    x_train, x_test, tag_train, tag_test = train_test_split(input_x, y, test_size=0.2, random_state=random_state)
    return  x_train, x_test, tag_train, tag_test


def train_and_predicted_with_graid(input, param_grid, cv=5):
    input_x, y = input
    scoring = ['precision_macro', 'recall_macro', 'f1_macro']
    clf = linear_model.LogisticRegressionCV(n_jobs=-1)
    grid = GridSearchCV(clf, param_grid, cv=cv, scoring='accuracy')
    scores = grid.fit(input_x, y)
    print('parameters:')
    best_parameters = grid.best_estimator_.get_params()
    for param_name in sorted(best_parameters):
        print('\t%s: %r' %(param_name, best_parameters[param_name]))
    return scores

# print('\t\tuse max_df,min_df=(1.0,0.0) to extract feature,then logistic regression:\t\t')
max_df = [0.2, 0.4, 0.5, 0.8, 1.0, 1.5, 5]
min_df = [0, 0.1, 0.2, 0.3, 0.4]
#for i in min_df:
    #print('With min_df=',i)
# tfidf_matrix =feature_extraction(data_str,'tfidf',max_df=1.0,min_df=i)
# input=[tfidf_matrix,tag_str]
# x_train, x_test, tag_train, tag_test = split_data(input)
# fitandpredicted(x_train, x_test, tag_train, tag_test)
# print('time uesed: %0.4fs' %(time() - t0))


# C= [0.1, 0.2, 0.5, 0.8, 1.5, 3, 5]
# fit_intercept=[True, False]
# penalty=['l1', 'l2']
# solver=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
# param_grid=dict(C=C, fit_intercept=fit_intercept, penalty=penalty, solver=solver)
# tfidf_matrix = feature_extraction(data_str, 'tfidf', max_df=1, min_df=0.0)
# input=[tfidf_matrix,tag_str]
# scores = train_and_predicted_with_graid(input, cv=5, param_grid=param_grid)


