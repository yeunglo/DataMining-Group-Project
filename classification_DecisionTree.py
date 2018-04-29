
# 1st class: v[1]
import os
import json
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
class Classification_JCL():
    def __init__(self):
        self.vec = []
        self.toxic = []
        # self.severe_toxic = []
        # self.obscene =[]
        # self.threat = []
        # self.insult = []
        # self.identity_hate = []
        self.words = []
        #self.key = []
        self.result = []
        self.classifier = None
    def read_training_set(self):
        with open('train_full.json') as json_train:
            train = json.load(json_train)
            for key,value in train.items():
                self.words.append(value[0])
                self.toxic.append(value[1])
                # self.severe_toxic.append(value[2])
                # self.obscene.append(value[3])
                # self.threat.append(value[4])
                # self.insult.append(value[5])
                # self.identity_hate.append(value[6])
                #self.key.append(key)
        json_train.close()
    def feature_matrix(self):
        Vector = DictVectorizer()
        self.vec = Vector.fit_transform(self.words).toarray()

    def tf_idf(self):
        files = os.listdir('./')
        saved_file_name = 'doc_matrix.pkl'
        if saved_file_name not in files:
            tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=250000,min_df=0.01, stop_words='english',
                                             use_idf=True, ngram_range=(1, 1))
            tfidf_matrix = tfidf_vectorizer.fit_transform(self.words) #fit the vectorizer to synopses
            joblib.dump(tfidf_matrix, saved_file_name)
        else:
            tfidf_matrix = joblib.load(saved_file_name)
        self.vec = tfidf_matrix

    def Decision_Tree(self):
        clf = tree.DecisionTreeClassifier()
        clf.fit(self.vec,self.toxic)
        self.result=clf.predict(self.vec)
        rate = 1-sum(abs(i) for i in (self.result-self.toxic))/len(self.toxic)
        print rate

    def write_result(self):
        jsObj = json.dumps(self.result)
        fileObject = open('result_full.json', 'w')
        fileObject.write(jsObj)
        fileObject.close()

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
    Test.read_training_set()
    Test.feature_matrix()
    #Test.tf_idf()
    Test.Decision_Tree()
    #Test.write_result()