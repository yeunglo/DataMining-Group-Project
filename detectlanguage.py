from langdetect import detect_langs
from langdetect import detect
import csv
results = 0
test_path = 'train1.csv'
path_full = 'C:\Users\home\Documents\Data Mining\\toxic comment\\train.csv'
with open(path_full) as Train:
    reader = csv.DictReader(Train)
    for line in reader:
        try:
            if detect(line['comment_text'].decode('utf-8')) != 'en':
                results +=1
        except:
            print('No features in text.')

print 'Number of non-en: '
print results

# str = ''
# if str !='':
#     print detect(str)
# else:
#     print 'oooh'



