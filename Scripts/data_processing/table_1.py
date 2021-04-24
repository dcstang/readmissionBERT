import os
import re
import pandas as pd
import numpy as np

path = os.getcwd()
work_dir = os.path.join(path, 'Sem 2 - Machine Learning/Project')

dfUadm = pd.read_csv(os.path.join(work_dir, 'Data/stripped_CLEANED_FULL_UADM.csv'),
      dtype={'SUBJECT_ID' : 'UInt32', 'HADM_ID' : 'UInt32', 'TEXT_CONCAT' : 'string',
         'ADMISSION_TYPE' : 'string', 'ETHNICITY' : 'string', 'DIAGNOSIS' : 'string',
         'HOSPITAL_EXPIRE_FLAG' : 'bool', 'MORTALITY_30D' : 'bool', 'TARGET' : 'bool',
         'AGE' : 'UInt8'},
      parse_dates=['ADMITTIME', 'DISCHTIME', 'NEXT_UADMITTIME', 'DOD_SSN'],
      header=0)

# punch out table 1
# split to cases and controls
dfCases = dfUadm[dfUadm.TARGET == 1]
dfControls = dfUadm[dfUadm.TARGET == 0]


# get N
len(dfUadm.SUBJECT_ID.unique())     #N = 32722
len(dfCases.SUBJECT_ID.unique())    #N = 2275
len(dfControls.SUBJECT_ID.unique()) #N = 32,311

# sex
dfUadm.groupby('GENDER').count()        #F = 18481  / M = 23984
dfCases.groupby('GENDER').count()       #F = 1275   / M = 1706
dfControls.groupby('GENDER').count()    #F = 17,206 / M = 22,278

# age 
dfUadm.AGE.mean()       # 62.7
dfCases.AGE.mean()      # 64.6
dfControls.AGE.mean()   # 62.5

dfUadm.AGE.std()        # 17.6
dfCases.AGE.std()       # 16.9
dfControls.AGE.std()    # 17.7

# ethnicity
dfUadm.groupby('ETHNICITY').count()
    # asian     1033
    # black     4238
    # hispanic  1596
    # white     30455
    # other     5143
dfCases.groupby('ETHNICITY').count()
    # asian     68
    # black     489
    # hispanic  122 
    # white     2118
    # other     184
dfControls.groupby('ETHNICITY').count()
    # asian     965
    # black     3749
    # hispanic  1474
    # white     28337
    # other     4959

# duration of stay
dfUadm['ADM_DURATION'] = dfUadm['DISCHTIME'] - dfUadm['ADMITTIME']
dfUadm['ADM_DURATION'] = dfUadm['ADM_DURATION'].dt.days

dfUadm['ADM_DURATION'].mean()       # 9.5 days
dfCases['ADM_DURATION'].mean()      # 12.2 days
dfControls['ADM_DURATION'].mean()   # 9.3 days

dfUadm['ADM_DURATION'].std()        # 10.4 days
dfCases['ADM_DURATION'].std()       # 13.1 days
dfControls['ADM_DURATION'].std()    # 10.2 days

dfUadm['ADM_DURATION'].median()     # 7 days
dfCases['ADM_DURATION'].median()    # 8 days
dfControls['ADM_DURATION'].median() # 6 days

#  diagnosis
dfUadm.DIAGNOSIS.mode()
dfUadm.DIAGNOSIS.value_counts()
    # pneumonia     1198
    # sepsis        852
    # CAD           810
    # CHF           752
dfCases.DIAGNOSIS.value_counts()
    # pneumonia     139
    # sepsis        96
    # CHF           107
dfControls.DIAGNOSIS.value_counts()
    # pneumonia     1059
    # sepsis        756
    # CAD           777
    # CHF           645

# do word counts
import spacy
import scispacy
from collections import Counter
from tqdm import tqdm
import itertools
import pickle

nlp = spacy.load('en_core_web_md', exclude=['ner', 'senter', 'lemmatizer', 'AttributeRuler'])

# remove these from stopwords, negatives important in medical text
nlp.Defaults.stop_words.remove('no')
nlp.Defaults.stop_words.remove('not')
nlp.max_length = 5000000

tokens = []
stops = []

for doc in nlp.pipe(dfUadm.TEXT_CONCAT.values):

    if doc.has_annotation("DEP"):
    # all tokens that arent stop words or punctuations
        tokens.append([token.text for token in doc
        if not token.is_stop and not token.is_punct])

    # stop words 
        stops.append([token.text for token in doc 
        if (token.is_stop and not token.is_punct)])


# five most common tokens >> for cases and controls
tokens = list(itertools.chain(*tokens))
word_freq = Counter(tokens)
common_words = word_freq.most_common(5)

# five most common noun tokens
stops = list(itertools.chain(*stops))
stop_freq = Counter(stops)
common_stops = stops.most_common(5)

# pickle counts for later use
with open(os.path.join(work_dir, "../datasplit/tokens.pkl"), "wb") as file:
    pickle.dump(tokens, file)

with open(os.path.join(work_dir, "../datasplit/stops.pkl"), "wb") as file:
    pickle.dump(stops, file)


# pandas implementation of counts
tokens = dfUadm.TEXT_CONCAT.str.split()
results = Counter()

dfUadm.TEXT_CONCAT.str.split().apply(results.update)
print(list(results.items())[:10])
sum(results.values()) # 106,175,268