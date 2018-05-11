import csv
import re
import os
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn import metrics
from time import time
import json
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

stemmer = SnowballStemmer("english")

def getwords(doc):
    splitter = re.compile('\\W+[0-9]*') # split with non-words
    stopworddic = set(stopwords.words('english'))
    words=[s.lower() for s in splitter.split(doc)
           if len(s)>2 and len(s)<20 and s not in stopworddic]
    stems = [stemmer.stem(t) for t in words]
    return stems

class lightgbm():
    def __init__(self):
        self.vec = []
        self.target = []
        self.words = []
        self.result = []
        self.train_data = None
        self.test_data = None
        self.train_target = None
        self.test_target = None

    def read_csv(self):
        path_full = 'C:\Users\home\Documents\Data Mining\\toxic comment\\train.csv'
        print('Load data...')
        #labels = ['id','content','toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        #
        # self.words = pd.read_csv(data,usecols= [1])
        # self.target = pd.read_csv(data, usecols= [2])
        with open(path_full) as Train:
            reader = csv.DictReader(Train)
            #labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
            for line in reader:
                self.words.append((line['comment_text']))
                self.target.append(float(line['toxic']))

    def tf_idf(self):
        files = os.listdir('./')
        saved_file_name = 'doc_matrix_full2.pkl'
        if saved_file_name not in files:
            tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=250000,min_df=0.01,tokenizer=getwords, stop_words='english',
                                             use_idf=True, ngram_range=(1, 1))
            tfidf_matrix = tfidf_vectorizer.fit_transform(self.words) #fit the vectorizer to synopses
            joblib.dump(tfidf_matrix, saved_file_name)
        else:
            tfidf_matrix = joblib.load(saved_file_name)
        self.vec = tfidf_matrix

    def split_data(self,X,y):
        print('Spliting dataset...')
        self.train_data, self.test_data, self.train_target, self.test_target = train_test_split(X, y, test_size=0.1, random_state=10, shuffle=True)

    def split_without_imbalance(self):
        print('Spliting dataset...')
        self.train_data, self.test_data, self.train_target, self.test_target = train_test_split(self.vec, self.target, test_size=0.2,
                                                                                                random_state=10,
                                                                                                shuffle=True)
    def balance_smote(self):
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_sample(self.vec, self.target)
        print ('SMOTE:')
        lightgbm.split_data(self,X_res,y_res)
    def balance_rus(self):
        print('RUS')
        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_sample(self.vec, self.target)
        print ('RandomUnderSampler:')
        lightgbm.split_data(self, X_res, y_res)
        #print self.train_data.shape

    def classifier(self):
        print('Modelling: ')
        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'min_data_in_leaf':50,
            'num_leaves': 30,
            'max_depth' : 20,
            'learning_rate': 0.1,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'bagging_freq': 5,
            'verbose': 0
        }
        lgb_train = lgb.Dataset(self.train_data,self.train_target)
        lgb_eval = lgb.Dataset(self.test_data,self.test_target,reference = lgb_train)


        gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5)


        print('start predicting with lightGBM')
        self.result = gbm.predict(self.test_data, num_iteration=gbm.best_iteration)
        print(metrics.classification_report(self.test_target, self.result.round()))
        #print('The rmse of prediction is:', mean_squared_error(self.test_target, self.result) ** 0.5)
if __name__ == '__main__':
    t0 = time()
    Test = lightgbm()
    Test.read_csv()
    Test.tf_idf()
    #Test.balance_smote()
    Test.split_without_imbalance()
    Test.classifier()
    print('time uesed: %0.4fs' % (time() - t0))