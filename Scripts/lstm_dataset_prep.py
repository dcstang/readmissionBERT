import pandas as pd
import numpy as np
import os
import re
from textaugment import EDA

from sklearn.model_selection import train_test_split 

path = os.getcwd() #.. mimic/script
work_dir = os.path.join(path, 'Sem 2 - Machine Learning/Project/Scripts')

dfUadm = pd.read_csv(os.path.join(work_dir, '../Data/lemma_dfUadm.csv'),
	  dtype={'SUBJECT_ID' : 'UInt32', 'HADM_ID' : 'UInt32', 'TEXT_CONCAT' : 'string',
		 'ADMISSION_TYPE' : 'string', 'ETHNICITY' : 'string', 'DIAGNOSIS' : 'string',
		 'HOSPITAL_EXPIRE_FLAG' : 'bool', 'MORTALITY_30D' : 'bool', 'TARGET' : 'bool',
		 'AGE' : 'UInt8'},
	  parse_dates=['ADMITTIME', 'DISCHTIME', 'NEXT_UADMITTIME', 'DOD_SSN'],
	  header=0)

def final_clean(text):
	text = re.sub(r"[^a-zA-Z\d\s:]", "", text)
	text = re.sub(r"\b[a-zA-Z]\b", "", text)
	text = re.sub(r"\b\w{1,4}\b", "", text)
	text = re.sub(r"\s\s+", " ", text)
	text = re.sub(r"(^(?:\S+\s+\n?){1,20})", "", text)
	return text

dfUadm['TEXT_CONCAT'] = dfUadm['TEXT_CONCAT'].apply(final_clean)
X = dfUadm.TEXT_CONCAT
Y = dfUadm.TARGET

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.15, random_state=77, stratify=Y)

print("Initial stratified test_train_split 0.15 and info about train_set:")
print("Number and prop(%) of cases   : ", (y_train == 1).sum(), 
			", % =", round((y_train == 1).sum()/len(y_train), 3))
print("Number and prop(%) of controls: ", (y_train == 0).sum(), 
			", % =", round((y_train == 0).sum()/len(y_train), 3))

repeatCases = len(y_train.index[y_train == True])
x_train = x_train.append([x_train[y_train[y_train==True].index]]*5)
y_train = y_train.append([y_train[y_train==True]]*5)

# text augmentation to replicated true cases
t = EDA() # leave one set of original cases
x_train[-repeatCases*2:-repeatCases] = x_train[-repeatCases*2:-repeatCases].apply(t.random_deletion, args=(0.20,))
x_train[-repeatCases*3:-repeatCases*2] = x_train[-repeatCases*4:-repeatCases*3].apply(t.random_deletion, args=(0.15,))
x_train[-repeatCases*4:-repeatCases*3] = (
	x_train[-repeatCases*4:-repeatCases*3]
	.apply(t.random_swap, args=(100,))
	.apply(t.random_deletion, args=(0.10,))
)
x_train[-repeatCases*5:-repeatCases*4] = (
	x_train[-repeatCases*5:-repeatCases*4]
	.apply(t.random_swap, args=(100,))
	.apply(t.random_insertion, args=(100,))
	.apply(t.random_deletion, args=(0.10,))
)

print("Post text augmentation train_set:")
print("Augmentations applied :", "\n    random deletion 15-20%,\n    random swap 100 words,\n    random insertion of 100 words")
print("Number and prop(%) of cases   : ", (y_train == 1).sum(), 
			", % =", round((y_train == 1).sum()/len(y_train), 3))
print("Number and prop(%) of controls: ", (y_train == 0).sum(), 
			", % =", round((y_train == 0).sum()/len(y_train), 3))

print("x_train dims: ", x_train.shape)
print("x_test dims : ", x_test.shape, "\n")


# save train test - save in text file  
# need to split by cases and controls

x_train_cases = x_train[y_train == 1]
x_train_controls = x_train[y_train == 0]

x_train_cases.to_csv(os.path.join(work_dir, '../Data/LSTM_data/train/1_train_cases.txt'), sep=' ', header=False, index=False)
x_train_controls.to_csv(os.path.join(work_dir, '../Data/LSTM_data/train/0_train_controls.txt'), sep=' ', header=False, index=False)


x_test_cases = x_test[y_test == 1]
x_test_controls = x_test[y_test == 0]

x_test_cases.to_csv(os.path.join(work_dir, '../Data/LSTM_data/test/1_test_cases.txt'), sep=' ', header=False, index=False)
x_test_controls.to_csv(os.path.join(work_dir, '../Data/LSTM_data/test/0_test_controls.txt'), sep=' ', header=False, index=False)



with open(os.path.join(DATA_DIR, 'train/1_train_cases.txt'), 'r') as f, open(os.path.join(DATA_DIR, 'train/train_cases.txt'), 'w') as fo:
	for line in f:
		fo.write(line.replace('"', '').replace("'", ""))