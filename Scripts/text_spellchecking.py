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

## compile a series of regex 
# cap number of consecutive newline characters to 2
newline_regex = re.compile(r'(\\n){3,}') 
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
routine = re.compile(r"Followup.Instructions:.*", re.DOTALL)

# cleaning functions 
def regex_cleaning(text):
    text = text.lower()
    text = date_regex.sub(r'\1/\2/\3',text)
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(routine, "", text)
    text = newline_regex.sub(r' \\n\\n ',text)
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

    # verbose feedback
    global i
    i += 1
    if (i % 1000) == 0:
        print(i)

    return tokenized_text 

# init
i = 0
num_tokens = list()
num_sents = list()

nlp.tokenizer.add_special_case(u'<PAR>', [{ORTH: u'<PAR>'}])
nlp.tokenizer.add_special_case(u'<UNK>', [{ORTH: u'<UNK>'}])


# 1. regex cleaning & tokenize with scispacy on pandas apply
# 2. add tokens to dictionary (<=3 words), only need to be done once 
# 3. spellchecker with levenshtein distance = 1

word_freq = Counter()
dfFull["TEXT_CONCAT"] = dfFull["TEXT_CONCAT"].apply(scispacy_tokenization, args = (word_freq,))

# freq and infreq words 
infreq_words = [word for word in word_freq.keys() if word_freq[word] <= 3 and word[0].isdigit() == False]
print(len(infreq_words))
sorted(infreq_words)[10000:10050]

freq_words = [word for word in word_freq.keys() if word_freq[word] > 3]
add_to_dictionary = " ".join(freq_words)
f=open(os.path.join(work_dir, "Data/mimic_dict.txt"), "w+")
f.write(add_to_dictionary)
f.close()

# find mispelled words among infrequent words from dictionary 
# create mispelling dictionary
spell = SpellChecker()
spell.distance = 1  # set the distance parameter to just 1 edit away
spell.word_frequency.load_text_file(os.path.join(work_dir, "Data/mimic_dict.txt"))

misspelled = spell.unknown(infreq_words)
misspell_dict = {}
for i, word in enumerate(misspelled):
    if (word != spell.correction(word)):
        misspell_dict[word] = spell.correction(word)

# inspect dictionary 
print(len(misspell_dict))
list(misspell_dict.items())[:30]

# tokenize remainding words <4 freq as UNK 
unk_words = [word for word in infreq_words if word not in list(misspell_dict.keys())]
print(len(unk_words))
unk_words[:100]

np.savetxt(os.path.join(work_dir, 'Data/discharge_unk_words.txt'), unk_words, fmt='%s', newline=os.linesep)
f=open(os.path.join(work_dir, "Data/discharge_typos.txt", "w+"))
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

counter = 0
def fix_typos(text):
    global counter
    
    found = processor.replace_keywords(text)
    
    counter+=1
    if (counter % 1000) == 0:
        print (counter)
    
    return found

dfFull["TEXT_CONCAT"] = dfFull["TEXT_CONCAT"].apply(fix_typos)


# save dataframe here 
dfFull.to_csv(os.path.join(work_dir, 'Data/CLEANED_FULL_UADM.csv'), index=False)