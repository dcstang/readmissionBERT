import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.tokenize.toktok import ToktokTokenizer

path = os.getcwd()
work_dir = os.path.join(path, 'Sem 2 - Machine Learning/Project')

# init nltk tokenizer 
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')

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

# pre-embedding feature engineering:  stopword removal, lemmatization 
def nltk(text):
    text = remove_stopwords(text)
    tokenized_text = lemmatize_text(text)
    return tokenized_text 