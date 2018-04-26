# -*- coding: UTF-8 -*-

import json
import csv
import nltk
# import enchant
import string
import re
import os
from config import Config as conf
from nltk.corpus import wordnet as wn
import sys
#nltk.download('punkt')
def getwords(doc):
    splitter = re.compile('\\W*')
    words=[s.lower() for s in splitter.split(doc)
           if len(s)>2 and len(s)<20]
    return dict([(w, words.count(w)) for w in words])
def CleanLines(line):
    identify = string.maketrans('', '')
    delEStr = string.punctuation +string.digits  #ASCII 标点符号，数字
    cleanLine = line.translate(identify,delEStr) #去掉ASCII 标点符号和空格
    cleanLine =line.translate(identify,delEStr) #去掉ASCII 标点符号
    return cleanLine
comment={}
#change the address as you want
with open('C:/Users/admin/Desktop/dm-group/train.csv') as Train:
    reader = csv.DictReader(Train)
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    for line in reader:
        #print(line['id'])
        id=line['id']
        com_label=[]
        stocks=CleanLines(line['comment_text'])
        print(stocks)
        com_label.append(getwords(stocks))
        for i in labels:
            com_label.append(int(line[i]))
        comment[id]=com_label
    #print(comment)

jsObj = json.dumps(comment)
#change the address as you want
fileObject = open('C:/Users/admin/Desktop/dm-group/train_original.json', 'w')
fileObject.write(jsObj)
fileObject.close()