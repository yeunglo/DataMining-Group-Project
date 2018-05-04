import operator
import csv
import re
import os
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

stemmer = SnowballStemmer("english")

def getwords(doc):
    splitter = re.compile('\\W+[0-9]*') # split with non-words
    stopworddic = set(stopwords.words('english'))
    words=[s.lower() for s in splitter.split(doc)
           if len(s)>2 and len(s)<20 and s not in stopworddic]
    stems = [stemmer.stem(t) for t in words]
    return dict([(w, stems.count(w)) for w in stems])

class Classification_JCL():
    def __init__(self):
        self.vec = []
        self.target = []
        self.words = []
        #self.key = []
        self.result = []
        self.classifier = None

    def read_csv(self):
        path_full = 'C:\Users\home\Documents\Data Mining\\toxic comment\\train.csv'
        with open(path_full) as Train:
            reader = csv.DictReader(Train)
            #labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

            for line in reader:
                self.words.append((line['comment_text']))
                self.target.append(float(line['identity_hate']))

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

        rate = sum(abs(i) for i in  map(operator.sub, self.target, self.result))/len(self.target)
        print 1.0-rate

        # def write_result(self):
    #     jsObj = json.dumps(self.result)
    #     fileObject = open('result_full.json', 'w')
    #     fileObject.write(jsObj)
    #     fileObject.close()

    def split_data(input, random_state=10, shuffle=True):
        input_x, y = input
        # print input_x
        x_train, x_test, tag_train, tag_test = train_test_split(input_x, y, test_size=0.2, random_state=random_state)
        return x_train, x_test, tag_train, tag_test

if __name__ == '__main__':
    Test = Classification_JCL()
    Test.read_csv()
    Test.tf_idf()
    #Test.split_data()
    Test.Decision_Tree()