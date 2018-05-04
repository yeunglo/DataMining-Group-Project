import datetime

starttime = datetime.datetime.now()

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from time import time

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

with open('train_original.json') as Train_data:
    train_data = json.load(Train_data)


# reader1={"78915629f8418c85": [{"and": 1, "hair": 1, "shaggy": 1, "also": 1, "very": 1, "some": 1,"dark": 1,"jewish": 1,"looks": 1,"imagine": 1,"you": 1,"beard": 1,"with": 1,"him": 1,"can": 1}, 1, 0, 0, 0, 0, 0], "ae6308347d82cb41": [{"about": 1, "think": 1, "your": 1, "hgbob": 1, "edit": 1, "please": 1, "some": 1, "here": 1, "you": 1, "may": 1, "see": 1, "flags": 1, "controversial": 1, "ask": 1, "the": 1, "debate": 1, "talk": 1, "theres": 1}, 1, 0, 0, 0, 0, 0], "bdf44cf93bd918a3": [{"redirect": 1, "despot": 1, "talkjohn": 1, "palaiologos": 1}, 0, 0, 0, 0, 0, 0], "291708466f6a09e4": [{"all": 1, "jimim": 1, "lout": 1}, 1, 1, 0, 0, 0, 0]}

def get_data(data):
    words = []
    labels_com = []
    wordslist = []
    for key in data:
        words.append(data[key][0])
        label_com = str(data[key][1])
        for i in range(2, 7):
            label_com += str(data[key][i])
        label_com = int(label_com, 2)
        labels_com.append(label_com)
    print(labels_com)
    for item in words:
        wordslist.append(str(" ".join(i for i in item)))
    print(wordslist[0:3])
    return wordslist, labels_com

def feature_extraction(data_str, case, max_df=1.0, min_df=0.0):
    tfidf_matrix = TfidfVectorizer(token_pattern='\w', ngram_range=(1, 2), max_df=max_df, min_df=min_df).fit_transform(
        data_str)
    return tfidf_matrix

def split_data(input, random_state=1, shuffle=True):
    input_x, y = input
    # print input_x
    x_train, x_test, tag_train, tag_test = train_test_split(input_x, y, test_size=0.1, random_state=random_state)
    return x_train, x_test, tag_train, tag_test

def knn(n, x_train, x_test, tag_train, tag_test):
    estimator = KNeighborsClassifier(n)
    print('estimator setup')
    estimator.fit(x_train, tag_train)
    print('estimator done')
    tag_predicted = estimator.predict(x_test)
    print('prediction done')
    accuracy = np.mean(tag_test == tag_predicted) * 100
    print('accuracy: {0: .3f}%'.format(accuracy))
    return accuracy

# print('\t\tuse max_df,min_df=(1.0,0.0) to extract feature,then logistic regression:\t\t')
max_df = [0.2, 0.4, 0.5, 0.8, 1.0, 1.5]
min_df = [0, 0.1, 0.2, 0.3, 0.4]

wordslist, labels_com = get_data(train_data)

'''
# find the best max and min value of tfidf.
best_df_accuracy = 0
for max in max_df:
    for min in min_df:
        if min < max:
            print('With max_df=', max)
            print('With min_df=', min)
            tfidf_matrix = feature_extraction(wordslist, 'tfidf', max_df=max, min_df=min)
            input = [tfidf_matrix, labels_com]
            x_train, x_test, tag_train, tag_test = split_data(input)
            accuracy = knn(30, x_train, x_test, tag_train, tag_test)
            if best_df_accuracy < accuracy:
                best_df_accuracy = accuracy
                best_max_df = max
                best_min_df = min

print(best_max_df, best_min_df, best_df_accuracy)
'''

tfidf_matrix = feature_extraction(wordslist, 'tfidf', max_df=0.4, min_df=0)
input = [tfidf_matrix, labels_com]
x_train, x_test, tag_train, tag_test = split_data(input)

'''
# find the best k value
best_k = 0
best_accuracy = 0
for n in range(1,51):
    accuracy = knn(n, x_train, x_test, tag_train, tag_test)
    if accuracy > best_accuracy:
        best_k = n
        best_accuracy = accuracy
print(best_k, best_accuracy)
'''

accuracy = knn(10, x_train, x_test, tag_train, tag_test)

endtime = datetime.datetime.now()
print((endtime - starttime).seconds)
