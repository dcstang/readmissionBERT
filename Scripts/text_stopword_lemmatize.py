import os
import re
import pandas as pd
import numpy as np
import scipy
import spacy
from tqdm import tqdm


path = os.getcwd()
work_dir = os.path.join(path, 'Sem 2 - Machine Learning/Project')

dfUadm = pd.read_csv(os.path.join(work_dir, 'Data/CLEANED_FULL_UADM.csv'),
      dtype={'SUBJECT_ID' : 'UInt32', 'HADM_ID' : 'UInt32', 'TEXT_CONCAT' : 'string',
         'ADMISSION_TYPE' : 'string', 'ETHNICITY' : 'string', 'DIAGNOSIS' : 'string',
         'HOSPITAL_EXPIRE_FLAG' : 'bool', 'MORTALITY_30D' : 'bool', 'TARGET' : 'bool',
         'AGE' : 'UInt8'},
      parse_dates=['ADMITTIME', 'DISCHTIME', 'NEXT_UADMITTIME', 'DOD_SSN'],
      header=0)

tqdm.pandas(desc="Pandas Apply Progress")

# Logistic Regression Preprocessing            
# full stripping of numeric entities and dash for logistic reg and SVM

def regex_strip_down(text):
	text = re.sub(r"<PAR>", "", text)
	text = re.sub(r"[\d.]", "", text)
	text = re.sub(r"-", "", text)
	text = re.sub(r"\d+", "", text)
	text = re.sub(r":", "", text)
	text = re.sub(r"\s+" , " ", text)
	text = re.sub(r"%", "", text)
	return text

def regex_strip_secondary(text):
	text = re.sub(r",", "", text)
	text = re.sub(r"//", "", text)
	text = re.sub(r"yo", "", text)
	text = re.sub(r"date", "", text)
	return text

def regex_question(text):
	text = re.sub(r"\?", "", text)
	return text

dfUadm['TEXT_CONCAT'] = dfUadm['TEXT_CONCAT'].progress_apply(regex_strip_down)
dfUadm['TEXT_CONCAT'] = dfUadm['TEXT_CONCAT'].progress_apply(regex_strip_secondary)
dfUadm['TEXT_CONCAT'] = dfUadm['TEXT_CONCAT'].progress_apply(regex_question)
dfUadm.to_csv(os.path.join(work_dir, 'Data/stripped_CLEANED_FULL_UADM.csv'), index=False)

dfUadm = pd.read_csv(os.path.join(work_dir, 'Data/stripped_CLEANED_FULL_UADM.csv'),
      dtype={'SUBJECT_ID' : 'UInt32', 'HADM_ID' : 'UInt32', 'TEXT_CONCAT' : 'string',
         'ADMISSION_TYPE' : 'string', 'ETHNICITY' : 'string', 'DIAGNOSIS' : 'string',
         'HOSPITAL_EXPIRE_FLAG' : 'bool', 'MORTALITY_30D' : 'bool', 'TARGET' : 'bool',
         'AGE' : 'UInt8'},
      parse_dates=['ADMITTIME', 'DISCHTIME', 'NEXT_UADMITTIME', 'DOD_SSN'],
      header=0)

# split 5 chunks
dfUadm.iloc[0:4246].to_csv(os.path.join(work_dir, 'Data/checkpoint/splithpc/df_1.csv'), index=False)
dfUadm.iloc[4246:8492].to_csv(os.path.join(work_dir, 'Data/checkpoint/splithpc/df_2.csv'), index=False)   #done
dfUadm.iloc[8492:12738].to_csv(os.path.join(work_dir, 'Data/checkpoint/splithpc/df_3.csv'), index=False)  #done
dfUadm.iloc[12738:16984].to_csv(os.path.join(work_dir, 'Data/checkpoint/splithpc/df_4.csv'), index=False)
dfUadm.iloc[16984:21230].to_csv(os.path.join(work_dir, 'Data/checkpoint/splithpc/df_5.csv'), index=False)
dfUadm.iloc[21230:25476].to_csv(os.path.join(work_dir, 'Data/checkpoint/splithpc/df_6.csv'), index=False)
dfUadm.iloc[25476:29722].to_csv(os.path.join(work_dir, 'Data/checkpoint/splithpc/df_7.csv'), index=False)
dfUadm.iloc[29722:33968].to_csv(os.path.join(work_dir, 'Data/checkpoint/splithpc/df_8.csv'), index=False)
dfUadm.iloc[33968:38214].to_csv(os.path.join(work_dir, 'Data/checkpoint/splithpc/df_9.csv'), index=False)
dfUadm.iloc[38214:42465].to_csv(os.path.join(work_dir, 'Data/checkpoint/splithpc/df_10.csv'), index=False)

# redo-second
dfUadm.iloc[0:2123].to_csv(os.path.join(work_dir, 'Data/checkpoint/splithpc/df2_1.csv'), index=False)
dfUadm.iloc[2123:4246].to_csv(os.path.join(work_dir, 'Data/checkpoint/splithpc/df2_2.csv'), index=False)
dfUadm.iloc[4246:6369].to_csv(os.path.join(work_dir, 'Data/checkpoint/splithpc/df2_3.csv'), index=False)
dfUadm.iloc[6369:8492].to_csv(os.path.join(work_dir, 'Data/checkpoint/splithpc/df2_4.csv'), index=False)

dfUadm.iloc[16984:19107].to_csv(os.path.join(work_dir, 'Data/checkpoint/splithpc/df2_5.csv'), index=False)
dfUadm.iloc[19107:21230].to_csv(os.path.join(work_dir, 'Data/checkpoint/splithpc/df2_6.csv'), index=False)
dfUadm.iloc[23353:25476].to_csv(os.path.join(work_dir, 'Data/checkpoint/splithpc/df2_7.csv'), index=False)
dfUadm.iloc[27599:29722].to_csv(os.path.join(work_dir, 'Data/checkpoint/splithpc/df2_8.csv'), index=False)
dfUadm.iloc[29722:31845].to_csv(os.path.join(work_dir, 'Data/checkpoint/splithpc/df2_9.csv'), index=False)
dfUadm.iloc[31845:33968].to_csv(os.path.join(work_dir, 'Data/checkpoint/splithpc/df2_10.csv'), index=False)

# redo-third
dfUadm.iloc[33968:35029].to_csv(os.path.join(work_dir, 'Data/checkpoint/splithpc/df3_1.csv'), index=False)
dfUadm.iloc[35029:36090].to_csv(os.path.join(work_dir, 'Data/checkpoint/splithpc/df3_2.csv'), index=False)
dfUadm.iloc[36090:37151].to_csv(os.path.join(work_dir, 'Data/checkpoint/splithpc/df3_3.csv'), index=False)
dfUadm.iloc[37151:38212].to_csv(os.path.join(work_dir, 'Data/checkpoint/splithpc/df3_4.csv'), index=False)
dfUadm.iloc[38212:39273].to_csv(os.path.join(work_dir, 'Data/checkpoint/splithpc/df3_5.csv'), index=False)
dfUadm.iloc[39273:40334].to_csv(os.path.join(work_dir, 'Data/checkpoint/splithpc/df3_6.csv'), index=False)
dfUadm.iloc[40334:41395].to_csv(os.path.join(work_dir, 'Data/checkpoint/splithpc/df3_7.csv'), index=False)
dfUadm.iloc[41395:42465].to_csv(os.path.join(work_dir, 'Data/checkpoint/splithpc/df3_8.csv'), index=False)
# till 42465

### nlp pipe -- data parallelization ###
###  scripts from hereon running HPC ###
###  #PBS - J 1-10                   ###

# get HPC array number
csvindex = int(os.environ['PBS_ARRAY_INDEX'])
work_dir = os.getcwd() # ~/mimic/script

df = pd.read_csv(os.path.join(work_dir, '../datasplit/df_{}.csv'.format(csvindex)),
      dtype={'SUBJECT_ID' : 'UInt32', 'HADM_ID' : 'UInt32', 'TEXT_CONCAT' : 'string',
         'ADMISSION_TYPE' : 'string', 'ETHNICITY' : 'string', 'DIAGNOSIS' : 'string',
         'HOSPITAL_EXPIRE_FLAG' : 'bool', 'MORTALITY_30D' : 'bool', 'TARGET' : 'bool',
         'AGE' : 'UInt8'},
      parse_dates=['ADMITTIME', 'DISCHTIME', 'NEXT_UADMITTIME', 'DOD_SSN'],
      header=0)

print('Loading ... ', os.path.join(work_dir, '../datasplit/df_{}.csv'.format(csvindex)))


# init spacy specifically for lemmatization
nlp = spacy.load('en_core_sci_scibert', disable=['parser', 'ner'])

# remove these from stopwords, negatives important in medical text
nlp.Defaults.stop_words.remove('no')
nlp.Defaults.stop_words.remove('not')
stopwords = nlp.Defaults.stop_words
lemmatizer = nlp.get_pipe("lemmatizer")

def lemmatize_pipe(doc):
	lemma_list = [token.lemma_ for token in doc if not token.is_stop] 
	return " ".join(lemma_list)

def preprocess_pipe(texts):
	preproc_pipe = []
	for doc in nlp.pipe(texts, n_process=12, batch_size=1000):
		preproc_pipe.append(lemmatize_pipe(doc))
	return preproc_pipe
		
# execute stopword removal and lemmatization
df['TEXT_CONCAT'] = preprocess_pipe(df['TEXT_CONCAT'])

# save first and merge later
df.to_csv(os.path.join(work_dir, '../lemmatized/df_lem_{}.csv'.format(csvindex)), index=False)

# download and merge 10 csvs into one
# TODO: readjust filenames 
arrayFile = [
	'df_lem_1.csv', 
	'df_lem_2.csv', 
	'df_lem_3.csv', 
	'df_lem_4.csv', 
	'df_lem_5.csv', 
	'df_lem_6.csv', 
	'df_lem_7.csv', 
	'df_lem_8.csv', 
	'df_lem_9.csv', 
	'df_lem_10.csv']
opened = []

for file in arrayFile:
	df = pd.read_csv(os.path.join(work_dir, '..Data/checkpoint/remmerge', file), index_col= None, header = 0)
	opened.append(df)

frame = pd.concat(opened, axis = 0, ignore_index = True)
frame.to_csv(os.path.join(work_dir, '..Data/lemmatized/lemma_stripped_CLEANED_FULL_UADM.csv'), index=False)