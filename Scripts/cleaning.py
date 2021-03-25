import pandas as pd
import numpy as np
import os
import re
from nltk.tokenize.toktok import ToktokTokenizer
import nltk
import spacy
import scispacy 

path = os.getcwd()
work_dir = os.path.join(path, 'Sem 2 - Machine Learning/Project')

dfFull = pd.read_csv(os.path.join(work_dir, 'Data/merged.csv'),
      dtype={'SUBJECT_ID' : 'UInt32', 'HADM_ID' : 'UInt32', 'TEXT_CONCAT' : 'string',
         'ADMISSION_TYPE' : 'string', 'ETHNICITY' : 'string', 'DIAGNOSIS' : 'string',
         'HOSPITAL_EXPIRE_FLAG' : 'bool', 'MORTALITY_30D' : 'bool', 'TARGET' : 'bool'},
      parse_dates=['ADMITTIME', 'DISCHTIME', 'NEXT_UADMITTIME', 'DOD_SSN'],
      header=0)

df_X = dfFull.TEXT_CONCAT
df_Y = dfFull.TARGET

# init spacy
nlp = spacy.load('en_core_web_lg')
nlp2 = spacy.load('en_core_sci_scibert', exclude=['ner'])

# init nltk tokenizer 
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')

def lowercase(x):
    return x.lower()

def targetted(phrase):
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    routine = re.compile(r"Followup.Instructions:.*", re.DOTALL)
    phrase = re.sub(routine, "", phrase)
    return phrase

def remove_stopwords(text):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

# more extensive stopword removal, lemmatization 
def nltk(text):
    text = lowercase(text)
    text = targetted(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text 

def spacy(text):
    text = lowercase(text)
    text = lemmatize_text(text)
    return text 

def spacy_2(text):
    text = nlp(text)
    return text 

def scispacy(text):
    text = targetted(text)
    text = nlp2(text)
    return text 