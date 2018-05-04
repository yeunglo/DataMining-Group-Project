
# 1st class: v[1]
import operator
import csv
import re
import os
import json
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import numpy as np
stemmer = SnowballStemmer("english")


def getwords(doc):
    splitter = re.compile('\\W+[0-9]*') # split with non-words
    #print(splitter)
    stopworddic = set(stopwords.words('english'))
    words=[s.lower() for s in splitter.split(doc)
           if len(s)>2 and len(s)<20 and s not in stopworddic]
    stems = [stemmer.stem(t) for t in words]
    return dict([(w, stems.count(w)) for w in stems])

class Classification_JCL():
    def __init__(self):
        self.vec = []
        self.target = []
        #self.toxic = []
        # self.severe_toxic = []
        # self.obscene =[]
        # self.threat = []
        # self.insult = []
        # self.identity_hate = []
        self.words = []
        #self.key = []
        self.result = []
        self.classifier = None
    # def read_training_set(self):
    #     with open('train_full.json') as json_train:
    #         train = json.load(json_train)
    #         for key,value in train.items():
    #             self.words.append(value[0])
    #             self.target.append(value[1])
    #             # self.severe_toxic.append(value[2])
    #             # self.obscene.append(value[3])
    #             # self.threat.append(value[4])
    #             # self.insult.append(value[5])
    #             # self.identity_hate.append(value[6])
    #             #self.key.append(key)
    #     json_train.close()
    def read_csv(self):
        comment = {}
        path_test = 'C:\Users\home\Documents\GitHub\DataMining-Group-Project\\train1.csv'
        path_full = 'C:\Users\home\Documents\Data Mining\\toxic comment\\train.csv'
        with open(path_full) as Train:
            reader = csv.DictReader(Train)
            #labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

            for line in reader:
                # id = line['\xef\xbb\xbfid']
                #id = line['id']
                self.words.append((line['comment_text']))
                self.target.append(float(line['identity_hate']))
        #print self.words

    def feature_matrix(self):
        Vector = DictVectorizer()
        self.vec = Vector.fit_transform(self.words).toarray()

    def tf_idf(self):
        files = os.listdir('./')
        saved_file_name = 'doc_matrix_full.pkl'
        if saved_file_name not in files:
            tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=250000,min_df=0.01,tokenizer=getwords, stop_words='english',
                                             use_idf=True, ngram_range=(1, 1))
            tfidf_matrix = tfidf_vectorizer.fit_transform(self.words) #fit the vectorizer to synopses
            joblib.dump(tfidf_matrix, saved_file_name)
        else:
            tfidf_matrix = joblib.load(saved_file_name)
        self.vec = tfidf_matrix

    def Decision_Tree(self):
        clf = tree.DecisionTreeClassifier()
        clf.fit(self.vec,self.target)
        self.classifier = clf
        result=clf.predict(self.vec)
        for i in result:
            self.result.append(float(i))

        #print (self.toxic)
        rate = sum(abs(i) for i in  map(operator.sub, self.target, self.result))/len(self.target)
        print 1.0-rate

    def plottree(self):
        # dot_data = tree.export_graphviz(self.classifier, out_file=None,
        #                                 feature_names=self.vec,
        #                                 class_names=self.target,
        #                                 filled=True, rounded=True,
        #                                 special_characters=True)
        dot_data = tree.export_graphviz(self.classifier, out_file=None)
        #graph = graphviz.Source(dot_data)
        #graph.render("identity_hate")
        # def write_result(self):
    #     jsObj = json.dumps(self.result)
    #     fileObject = open('result_full.json', 'w')
    #     fileObject.write(jsObj)
    #     fileObject.close()

    # def Ten_fold_Validation(self):
    #     N = len(self.key)
    #     for i in range(0,10):
    #         self.testset = self.vec[i*(N/10):(i+1)*(N/10)]
    #         self.trainset = self.vec

    #         self.target = self.toxic[i*(N/10):(i+1)*(N/10)]
    #         self.classifier = clf.fit(self.train, self.target)
    #         self.result = self.classifier.predict(self.testset)



if __name__ == '__main__':
    Test = Classification_JCL()
    Test.read_csv()
    #Test.feature_matrix()
    Test.tf_idf()
    Test.Decision_Tree()
    #Test.plottree()
    #Test.write_result()


# 1st class: v[1]
import json
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
class Classification_JCL():
    def __init__(self):
        self.vec = []
        self.toxic = []
        self.severe_toxic = []
        self.obscene =[]
        self.threat = []
        self.insult = []
        self.identity_hate = []
        self.words = []
        # self.key = []
        self.testset=[]
        self.result = []
    def read_test_set(self):
        with open('test_original.json') as json_test:
            test = json.load(json_test)
            for key,value in test.items():
                self.testset.append(value[0])

    def read_training_set(self):
        with open('train_original.json') as json_train:
            train = json.load(json_train)
            for key,value in train.items():
                self.words.append(value[0])
                self.toxic.append(value[1])
                self.severe_toxic.append(value[2])
                self.obscene.append(value[3])
                self.threat.append(value[4])
                self.insult.append(value[5])
                self.identity_hate.append(value[6])
                # self.key.append(key)
    def feature_matrix(self):
        Vector = DictVectorizer()
        self.vec = Vector.fit_transform(self.words).toarray()

    def Decision_Tree(self):
        clf = tree.DecisionTreeClassifier()
        clf.fit(self.vec,self.toxic)
        self.result = clf.predict(self.testset)

    def write_result(self):
        jsObj = json.dumps(self.result)
        fileObject = open('result_original.json', 'w')
        fileObject.write(jsObj)
        fileObject.close()

    def Test(self):
        print self.testset

    # def PCA(self):

if __name__ == '__main__':
    Test = Classification_JCL()
    Test.read_training_set()
    Test.feature_matrix()
    Test.Decision_Tree()
    Test.write_result()
    # Test.Test()

