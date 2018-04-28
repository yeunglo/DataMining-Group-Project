
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
