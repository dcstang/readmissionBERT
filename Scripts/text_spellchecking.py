import pandas as pd
import numpy as np
import os
import re
import spacy
from spacy.symbols import ORTH
import scispacy 
from collections import Counter
from spellchecker import SpellChecker
from flashtext import KeywordProcessor
import nltk
from tqdm import tqdm
import pickle

path = os.getcwd()
work_dir = os.path.join(path, 'Sem 2 - Machine Learning/Project')

dfFull = pd.read_csv(os.path.join(work_dir, 'Data/merged.csv'),
      dtype={'SUBJECT_ID' : 'UInt32', 'HADM_ID' : 'UInt32', 'TEXT_CONCAT' : 'string',
         'ADMISSION_TYPE' : 'string', 'ETHNICITY' : 'string', 'DIAGNOSIS' : 'string',
         'HOSPITAL_EXPIRE_FLAG' : 'bool', 'MORTALITY_30D' : 'bool', 'TARGET' : 'bool',
         'AGE' : 'UInt8'},
      parse_dates=['ADMITTIME', 'DISCHTIME', 'NEXT_UADMITTIME', 'DOD_SSN'],
      header=0)


# init spacy
nlp = spacy.load('en_core_sci_scibert', exclude=['ner'])
tqdm.pandas(desc="Pandas Apply Progress")

## compile a series of regex 
# cap number of consecutive newline characters to 2
newline_regex = re.compile(r'(\\n){3,}') 
newline_regex2 = re.compile(r'(\\r){3,}')
ellipsis_regex = re.compile(r'(\.){2,}')
tilda_mult_regex = re.compile(r'(~){2,}')
atsign_mult_regex = re.compile(r'(@){2,}')
bracket_regex = re.compile(r'(.)(\()(.)')
bracket_regex2 = re.compile(r'(.)(\))(.)')
slash_regex = re.compile(r'(.)(\/)([^0-9])')
slash_regex2 = re.compile(r'([^0-9])(\/)(.)')
equals_regex = re.compile(r'(.)(=)(.)')
colon_regex = re.compile(r'(.)(:)(.)')
sq_bracket_regex = re.compile(r'(.)(\[)(.)')
dash_regex = re.compile(r'(.)(-)(.)')
dash_regex2 = re.compile(r'(-)([\S])')
plus_regex = re.compile(r'(.)(\+)(.)')
amp_regex = re.compile(r'(.)(&)(.)')
star_regex = re.compile(r'(.)(\*)(.)') 
comma_regex = re.compile(r'(.)(,)(.)')
tilda_regex = re.compile(r'(.)(~)(.)')
pipe_regex = re.compile(r'(.)(\|)(.)')
atsign_regex = re.compile(r'(.)(@)(.)')
dot_regex = re.compile(r'([^.][^0-9])(\.)([^0-9,][^.])')
dot_regex2 = re.compile(r'([^0-9])(\.)(.)')
semicol_regex = re.compile(r'(.);(.)')
caret_regex = re.compile(r'(.)\^(.)')
date_regex = re.compile(r'([0-9])-([0-9][0-9]?)-([0-9])')
routine = re.compile(r"followup.instructions*", re.DOTALL)

# cleaning functions 
def regex_cleaning(text):
    text = text.lower()
    text = text.replace("[**","[").replace("**]","]")
    text = date_regex.sub(r'\1/\2/\3',text)
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(routine, "", text)
    text = newline_regex.sub(r' \\n\\n ',text)
    text = newline_regex2.sub(r' \\n\\n ',text)
    text = ellipsis_regex.sub(r'.',text)
    text = tilda_mult_regex.sub(r'~',text)
    text = atsign_mult_regex.sub(r'@',text)
    
    text = text.replace("\n"," <PAR> ")
    text = bracket_regex.sub(r'\1 \2 \3',text)
    text = bracket_regex2.sub(r'\1 \2 \3',text)
    text = slash_regex.sub(r'\1 \2 \3',text)
    text = slash_regex2.sub(r'\1 \2 \3',text)
    text = slash_regex.sub(r'\1 \2 \3',text)
    text = equals_regex.sub(r'\1 \2 \3',text)
    text = colon_regex.sub(r'\1 \2 \3',text)
    text = sq_bracket_regex.sub(r'\1 \2 \3',text)
    text = dash_regex.sub(r'\1 \2 \3',text)
    text = dash_regex.sub(r'\1 \2 \3',text)
    text = dash_regex.sub(r'\1 \2 \3',text)
    text = dash_regex2.sub(r'\1 \2',text)
    text = plus_regex.sub(r'\1 \2 \3',text)
    text = star_regex.sub(r'\1 \2 \3',text)
    text = amp_regex.sub(r'\1 \2 \3',text)
    text = comma_regex.sub(r'\1 \2 \3',text)
    text = dot_regex.sub(r'\1 \2 \3',text)
    text = atsign_regex.sub(r'\1 \2 \3',text)
    text = tilda_regex.sub(r'\1 \2 \3',text)
    text = pipe_regex.sub(r'\1 \2 \3',text)
    text = dot_regex2.sub(r'\1 \3',text)
    text = semicol_regex.sub(r'\1 \2',text)
    text = caret_regex.sub(r'\1 \2',text)

    return text


# scispacy tokenization
def scispacy_tokenization(text, counter):
    
    text = regex_cleaning(text)
    tokens = nlp.tokenizer(text)
    tokenised_text = ""
    
    for token in tokens:
        tokenised_text = tokenised_text + token.text + " "
    
    counter.update(tokenised_text.split())
    tokenized_text = nltk.sent_tokenize(text)

    num_tokens.append(len(list(tokens)))
    num_sents.append(len(tokenized_text))

    return tokenized_text 

# init
num_tokens = list()
num_sents = list()

nlp.tokenizer.add_special_case(u'<PAR>', [{ORTH: u'<PAR>'}])
nlp.tokenizer.add_special_case(u'<UNK>', [{ORTH: u'<UNK>'}])


# 1. regex cleaning & tokenize with scispacy on pandas apply
# 2. add tokens to dictionary (<=3 words), only need to be done once 
# 3. spellchecker with levenshtein distance = 1

word_freq = Counter()
dfFull["TEXT_CONCAT"] = dfFull["TEXT_CONCAT"].progress_apply(scispacy_tokenization, args = (word_freq,))

with open(os.path.join(work_dir, 'Data/checkpoint/word_freq.pkl'), 'wb') as outputfile:
    pickle.dump(word_freq, outputfile)

# freq and infreq words 
infreq_words = [word for word in word_freq.keys() if word_freq[word] <= 3 and word[0].isdigit() == False]
print(len(infreq_words)) #102,784
# sorted(infreq_words)[10000:10050]

freq_words = [word for word in word_freq.keys() if word_freq[word] > 3]
add_to_dictionary = " ".join(freq_words) #1,162,524
f=open(os.path.join(work_dir, "Data/checkpoint/mimic_dict.txt"), "w+")
f.write(add_to_dictionary)
f.close()

# find mispelled words among infrequent words from dictionary 
# create mispelling dictionary
spell = SpellChecker()
spell.distance = 1  # set the distance parameter to just 1 edit away
spell.word_frequency.load_text_file(os.path.join(work_dir, "Data/checkpoint/mimic_dict.txt"))

misspelled = spell.unknown(infreq_words)
misspell_dict = {}
for i, word in enumerate(misspelled):
    if (word != spell.correction(word)):
        misspell_dict[word] = spell.correction(word)

# inspect dictionary 
print(len(misspell_dict)) #66,635
list(misspell_dict.items())[:30]

# tokenize remainding words <4 freq as UNK 
unk_words = [word for word in infreq_words if word not in set(misspell_dict.keys())]
print(len(unk_words))
unk_words[:100]

np.savetxt(os.path.join(work_dir, 'Data/discharge_unk_words.txt'), unk_words, fmt='%s', newline=os.linesep)
f=open(os.path.join(work_dir, "Data/discharge_typos.txt"), "w+")
for key in misspell_dict:
    f.write(key + '\t' + misspell_dict[key] + '\n')
f.close()

# spellcheck correction
keywords = list(misspell_dict.keys())
clean = list(misspell_dict.values())

processor = KeywordProcessor()

for keyword_name, clean_name in zip(keywords, clean):
    processor.add_keyword(keyword_name, clean_name)
    
for unk in unk_words:
    processor.add_keyword(unk, "<UNK>")


def flatten_list(text):
    return ''.join(text)    

def fix_typos(text):   
    found = processor.replace_keywords(text)    
    return found


dfFull["TEXT_CONCAT"] = dfFull["TEXT_CONCAT"].progress_apply(flatten_list)
dfFull["TEXT_CONCAT"] = dfFull["TEXT_CONCAT"].progress_apply(fix_typos)


# save dataframe here 
dfFull.to_csv(os.path.join(work_dir, 'Data/spellchecked_FULL_UADM.csv'), index=False)

# remove square brackets 
# remove ( namepattern2 ) and  'first name8'
# standardize pharmacy drug frequencies and route 
# remove annoying repeats ie. five ( 5 ) => five, qd ( daily ) => qd
# as above qid ( 4 times a day ) = qid
# ie. by mouth => po, intravenously => iv, subcutaneously => sc
# twice per day / b. id / q12h => bid, once per day => qd
# spacing p o = po, p r.n = prn
# account for <PAR> in between all above


def regex_pharmacy_clean(text):
    text = text.replace("[","").replace("]","")    
    text = re.sub(r"\((.*?)\)", "", text)
    text = re.sub(r"once daily", "qd", text)
    text = re.sub(r"once <PAR> daily", "qd", text)
    text = re.sub(r"once a day", "qd", text)
    text = re.sub(r"once <PAR> a day", "qd", text)
    text = re.sub(r"once a <PAR> day", "qd", text)
    text = re.sub(r"q\.*\s*d", "qd", text)

    text = re.sub(r"twice daily", "bid", text)
    text = re.sub(r"twice <PAR> daily", "bid", text)
    text = re.sub(r"twice a day", "bid", text)
    text = re.sub(r"two times a day", "bid", text)
    text = re.sub(r"two times daily", "bid", text)
    text = re.sub(r"2 times a day", "bid", text)
    text = re.sub(r"2 times daily", "bid", text)
    text = re.sub(r"twice a <PAR> day", "bid", text)
    text = re.sub(r"twice <PAR> a day", "bid", text)
    text = re.sub(r"b\.*\s*i\.*\s*d", "bid", text)
    
    text = re.sub(r"thrice daily", "tds", text)
    text = re.sub(r"thrice <PAR> daily", "tds", text)
    text = re.sub(r"thrice a day", "tds", text)
    text = re.sub(r"three times a day", "tds", text)
    text = re.sub(r"three times daily", "tds", text)
    text = re.sub(r"3 times a day", "tds", text)
    text = re.sub(r"3 times daily", "tds", text)
    text = re.sub(r"thrice a <PAR> day", "tds", text)
    text = re.sub(r"thrice <PAR> a day", "tds", text)
    text = re.sub(r"tid", "tds", text)
    text = re.sub(r"t\.*\s*i\.*\s*d", "tds", text)

    text = re.sub(r"four times daily", "qid", text)
    text = re.sub(r"4 times daily", "qid", text)
    text = re.sub(r"four times <PAR> daily", "qid", text)
    text = re.sub(r"four times a day", "qid", text)
    text = re.sub(r"4 times a day", "qid", text)
    text = re.sub(r"four times a <PAR> day", "qid", text)
    text = re.sub(r"four times <PAR> a day", "qid", text)
    text = re.sub(r"qds", "qid", text)
    text = re.sub(r"q\.*\s*i\.*\s*d", "qid", text)

    text = re.sub(r"first name.", "", text)
    text = re.sub(r"last name.", "", text)
    text = re.sub(r"name.", "", text)
    
    text = re.sub(r"intravenousl*y*", "iv", text)
    text = re.sub(r"subcutaneousl*y*", "sc", text)
    text = re.sub(r"by mouth", "po", text)
    text = re.sub(r"\bp\.*\s*o\.*\s*\b", "po ", text) #TODO: fix this for positive, potassium
    text = re.sub(r"\bp\.*\s*r\.*\s*n\b", "prn", text)

    return text

def clean_symbols(text):
    text = re.sub(r"\*", "", text)
    text = re.sub(r"\s/", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"=", "", text)
    return text

dfFull["TEXT_CONCAT"] = dfFull["TEXT_CONCAT"].progress_apply(regex_pharmacy_clean)
dfFull["TEXT_CONCAT"] = dfFull["TEXT_CONCAT"].progress_apply(clean_symbols)

dfFull.to_csv(os.path.join(work_dir, 'Data/CLEANED_FULL_UADM.csv'), index=False)