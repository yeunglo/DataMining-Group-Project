
# coding: utf-8
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk import SnowballStemmer
import re, string
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from pandas.core.frame import DataFrame

df = pd.read_csv('dataset/train.csv')
df1 = pd.read_csv(('dataset/test.csv'))
df.comment_text = df.comment_text.fillna('')
df1.comment_text = df1.comment_text.fillna('')

stops = set(stopwords.words("english"))
stops2 = set(stopwords.words("french"))
stops3 = set(stopwords.words("spanish"))
train = df.comment_text
test = df1.comment_text
special_character_removal = re.compile(r'[^a-z\d ]', re.IGNORECASE)
replace_numbers = re.compile(r'\d+',re.IGNORECASE)

def remove(text):
    text = text.lower().split()
    text = [w for w in text if not w in stops]
    text = [w for w in text if not w in stops2]
    text = [w for w in text if not w in stops3]
    text = " ".join(text)
    text = special_character_removal.sub('', text)
    text = replace_numbers.sub('n', text)

    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    return text

trainafter = []
for text in train:
    a = remove(text)
    trainafter.append(a)
testafter = []
for text in test:
    b = remove(text)
    testafter.append(b)
    
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()
cv = TfidfVectorizer(ngram_range=(2,2),tokenizer=tokenize,min_df=3,max_df=0.9,strip_accents='unicode',
                     use_idf=1,smooth_idf=1,sublinear_tf=1)
cv.fit(trainafter)
train_data = cv.transform(trainafter)
test_data = cv.transform(testafter)

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

submission = pd.DataFrame.from_dict({'id': df1['id']})
score = []
accuracy = []
precision = []
av_precision = []
recall = []
f1micro = []
f1macro = []
fmeasure = []
for label in  class_names:
    train_label = df[label].astype('int')
    mnb = MultinomialNB(alpha = 0.01)
    cv_score = np.mean(cross_val_score(mnb, train_data, train_label, cv=3,scoring='roc_auc'))
    #score.append(cv_score)
    accuracy_e = np.mean(cross_val_score(mnb, train_data, train_label, cv=3,scoring='accuracy'))
    accuracy.append(accuracy_e)
    pre = np.mean(cross_val_score(mnb, train_data, train_label, cv=3,scoring='precision'))
    precision.append(pre)
    av = np.mean(cross_val_score(mnb, train_data, train_label, cv=3,scoring='average_precision'))
    av_precision.append(av)
    recall_e = np.mean(cross_val_score(mnb, train_data, train_label, cv=3,scoring='recall'))
    recall.append(recall_e)
    f1 = np.mean(cross_val_score(mnb, train_data, train_label, cv=3,scoring='f1_micro'))
    f1micro.append(f1)
    f2 = np.mean(cross_val_score(mnb, train_data, train_label, cv=3,scoring='f1_macro'))
    f1macro.append(f2)
    fm = np.mean(cross_val_score(mnb, train_data, train_label, cv=3,scoring='f1'))
    fmeasure.append(fm)
    mnb.fit(train_data,train_label)
    #print(label,cv_score)
    submission[label] = mnb.predict_proba(test_data)[:,1]
    score.append(cv_score)

submission.to_csv('submission.csv',index=False)

d = {
    'Label': class_names,
    'F-measure': fmeasure,
    'F-measure(micro)':f1micro,
    'F-measure(macro)':f1macro,
    'ROC_AUC': score,
    'Precision': precision,
    'Recall': recall
}
e = {
    'Label': class_names,
    'Accuracy of each class': accuracy
}
result1 = DataFrame(e)
col = ['Label', 'Accuracy of each class']
result1 = result1.ix[:, col]
result2 = DataFrame(d)
cols = ['Label','Precision', 'Recall','F-measure', 'F-measure(macro)', 'F-measure(micro)', 'ROC_AUC']
result2 = result2.ix[:, cols]
result2

want = False
if want == True:
    acc = pd.read_csv('submission.csv')
    #proba = acc.toxic
    def zero(acc):
        zero = []
        c = 0
        for i in acc:
            if i>0.9:
                i=1
            else:
                i=0
            zero.append(i)
        return zero
    
    fsubmission = pd.DataFrame.from_dict({'id': df1['id']})
    for label in class_names:
        fsubmission[label] = zero(acc[label])
        fsubmission.to_csv('fsubmission.csv',index=False)
        

