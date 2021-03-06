
import csv
import re
import os
from sklearn.externals import joblib
from sklearn import tree
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
from time import clock
stemmer = SnowballStemmer("english")

def getwords(doc):
    splitter = re.compile('\\W+[0-9]*') # split with non-words

    stopworddic = set(stopwords.words('english'))
    words=[s.lower() for s in splitter.split(doc)
           if len(s)>2 and len(s)<20 and s not in stopworddic]
    stems = [stemmer.stem(t) for t in words]
    return stems

class Classification_JCL():
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

    def Dimension_Reduction(self):
        files = os.listdir('./')
        file_name = 'pca_matrix_full.pkl'
        if file_name not in files:
            pca = PCA(n_components=100, whiten=True, random_state=0)
            pca_matrix = pca.fit_transform(self.vec.toarray())
            joblib.dump(pca_matrix,file_name)
        else:
            pca_matrix = joblib.load(file_name)
        self.vec = pca_matrix

    def Decision_Tree(self):

        clf = tree.DecisionTreeClassifier()
        clf.fit(self.train_data,self.train_target)
        result=clf.predict(self.test_data)
        for i in result:
            self.result.append(float(i))

        # rate = sum(abs(i) for i in  map(operator.sub, self.test_target, self.result))/len(self.test_target)
        # print 'Accuracy: '+ str(1.0-rate)
        # TP = 0.0
        # FP = 0.0
        # TN = 0.0
        # FN = 0.0
        #
        # for i in range(0,len(self.test_target)-1):
        #     if (self.test_target[i] == 1 and self.result[i] == 1):
        #         TP += 1
        #     elif (self.test_target[i] == 0 and self.result[i] == 0):
        #         TN += 1
        #     elif (self.test_target[i] == 1 and self.result[i] == 0):
        #         FN += 1
        #     else:
        #         FP += 1
        # print 'Sensitivity: ' + str(TP/(TP+FN))
        # print 'Specificity: ' + str(TN/(TN+FP))
        print(metrics.classification_report(self.test_target, self.result))


    def split_data(self,X,y):
        self.train_data, self.test_data, self.train_target, self.test_target = train_test_split(X, y, test_size=0.1, random_state=10, shuffle=True)

    def balance_smote(self):
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_sample(self.vec, self.target)
        print ('SMOTE:')
        Classification_JCL.split_data(self,X_res,y_res)

    def balance_cnn(self):
        cnn = CondensedNearestNeighbour(random_state=42)
        X_res, y_res = cnn.fit_sample(self.vec, self.target)
        Classification_JCL.split_data(self, X_res, y_res)

    def balance_rus(self):
        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_sample(self.vec, self.target)
        print ('RandomUnderSampler:')
        Classification_JCL.split_data(self, X_res, y_res)

if __name__ == '__main__':
    clock()
    Test = Classification_JCL()
    Test.read_csv()
    Test.tf_idf()
    #Test.Dimension_Reduction()
    Test.balance_smote()
    #Test.balance_cnn()
    #Test.balance_rus()
    Test.Decision_Tree()
    print("running time is %-5.5ss" % clock())

