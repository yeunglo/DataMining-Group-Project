#!python3

import re
import csv
import json
import nltk
from nltk.corpus import stopwords


def getwords(doc):
    splitter = re.compile('\\W+[0-9]*') # split with non-words
    print(splitter)
    stopworddic = set(stopwords.words('english'))
    words=[s.lower() for s in splitter.split(doc)
           if len(s)>2 and len(s)<20 and s not in stopworddic]
    return dict([(w, words.count(w)) for w in words])

comment={}

with open('train1.csv') as Train:
    reader = csv.DictReader(Train)
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    for line in reader:
        #print(line['id'])
        id=line['\xef\xbb\xbfid']
        com_label=[]
        com_label.append(getwords(line['comment_text']))
        for i in labels:
            com_label.append(int(line[i])) # get values of six labels
        comment[id]=com_label 
    print(comment)


# output as json file

jsObj = json.dumps(comment)

fileObject = open('train_original.json', 'w')
fileObject.write(jsObj)
fileObject.close()





