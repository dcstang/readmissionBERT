import pandas as pd
import numpy as np
import scipy
import os
import re
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import spacy
import scispacy
from tqdm import tqdm
import eli5
import pickle 
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, \
					confusion_matrix, precision_score, recall_score,  \
					f1_score, roc_curve, auc
import xgboost as xgb

import tensorflow as tf
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

path = os.getcwd()
work_dir = os.path.join(path, 'Sem 2 - Machine Learning/Project')

dfUadm = pd.read_csv(os.path.join(work_dir, 'Data/CLEANED_FULL_UADM.csv'),
      dtype={'SUBJECT_ID' : 'UInt32', 'HADM_ID' : 'UInt32', 'TEXT_CONCAT' : 'string',
         'ADMISSION_TYPE' : 'string', 'ETHNICITY' : 'string', 'DIAGNOSIS' : 'string',
         'HOSPITAL_EXPIRE_FLAG' : 'bool', 'MORTALITY_30D' : 'bool', 'TARGET' : 'bool',
         'AGE' : 'UInt8'},
      parse_dates=['ADMITTIME', 'DISCHTIME', 'NEXT_UADMITTIME', 'DOD_SSN'],
      header=0)


########################### reconsider fasttext ###########################
# init nltk tokenizer 
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')

nlp = spacy.load('en_core_sci_scibert', exclude=['ner'])

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
def lemmatization_stopword(text):
    text = remove_stopwords(text)
    tokenized_text = lemmatize_text(text)
    return tokenized_text

# execute final clean before tf-idf
tqdm.pandas(desc="Pandas Apply Progress")
dfUadm['TEXT_CONCAT'] = dfUadm['TEXT_CONCAT'].progress_apply(lemmatization_stopword) ##?? too long 


#############################################################################

# pad to longest document



# Calculate the length of our vocabulary
word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(train_tweets)

vocab_length = len(word_tokenizer.word_index) + 1
vocab_length

def show_metrics(pred_tag, y_test):
    print("F1-score: ", f1_score(pred_tag, y_test))
    print("Precision: ", precision_score(pred_tag, y_test))
    print("Recall: ", recall_score(pred_tag, y_test))
    print("Acuracy: ", accuracy_score(pred_tag, y_test))
    print("-"*50)
    print(classification_report(pred_tag, y_test))
    
def embed(corpus): 
    return word_tokenizer.texts_to_sequences(corpus)


longest_train = max(texts, key=lambda sentence: len(word_tokenize(sentence)))
length_long_sentence = len(word_tokenize(longest_train))

train_padded_sentences = pad_sequences(
    embed(texts), 
    length_long_sentence, 
    padding='post'
)

embeddings_dictionary = dict()
embedding_dim = 100

# Load GloVe 100D embeddings
with open('/kaggle/input/glove6b100dtxt/glove.6B.100d.txt') as fp:
    for line in fp.readlines():
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary [word] = vector_dimensions

# embeddings_dictionary

# Now we will load embedding vectors of those words that appear in the
# Glove dictionary. Others will be initialized to 0.

embedding_matrix = np.zeros((vocab_length, embedding_dim))

for word, index in word_tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
        
embedding_matrix


## TODO: implement countvectorizer and Tfidf
X = dfUadm.TEXT_CONCAT
Y = dfUadm.TARGET

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.15, random_state=77, stratify=Y)
print(len(x_train), len(y_train))
print(len(x_test), len(y_test))


# instantiate the vectorizer
vect_tunned = CountVectorizer(stop_words='english', ngram_range=(1,2), 
    lowercase=False, min_df=0.1, max_df=0.7, max_features=10000)
# vect = CountVectorizer()
vect_tunned.fit(x_train)

with open(os.path.join(work_dir, 'Data/checkpoint/CountVect'), 'wb') as fout:
    pickle.dump(vect_tunned, fout)

# create a document-term matrix from train and test sets
x_train_dtm = vect_tunned.transform(x_train)
x_test_dtm = vect_tunned.transform(x_test)

"""
# checkpoint save sparse matrix
scipy.sparse.save_npz(os.path.join(work_dir, 'Data/checkpoint/x_train.npz'), x_train_dtm)
scipy.sparse.save_npz(os.path.join(work_dir, 'Data/checkpoint/x_test.npz'), x_test_dtm)
"""

# load sparse matrix checkpoint
x_train_dtm = scipy.sparse.load_npz(os.path.join(work_dir, 'Data/checkpoint/x_train.npz'))
x_test_dtm = scipy.sparse.load_npz(os.path.join(work_dir, 'Data/checkpoint/x_test.npz'))

# TODO: implement gridsearchCV for various C, add balanced weights
logit = LogisticRegression(C=0.01, solver='saga', multi_class='auto', 
                            random_state=17, n_jobs=4, max_iter=300, penalty='l1')
logit.fit(x_train_dtm, y_train)


logit_coef = eli5.show_weights(estimator=logit,
                feature_names=list(vect_tunned.get_feature_names()),
                top=(20,20))

logit_coef_all = eli5.show_weights(estimator=logit,
                feature_names=list(vect_tunned.get_feature_names()),
                top=None)

eli5.format_as_text(eli5.explain_weights(estimator=logit,
                feature_names=list(vect_tunned.get_feature_names()),
                top=(20,20)))

with open(os.path.join(work_dir, "Models/Logreg/top_20.html"), "wb") as file:
    file.write(logit_coef.data.encode("UTF-8"))

with open(os.path.join(work_dir, "Models/Logreg/all_coef.html"), "wb") as file:
    file.write(logit_coef_all.data.encode("UTF-8"))


def evaluation_plotting(model):

	y_pred_classes = model.predict(x_test_dtm)
	y_pred = (model.predict_proba(x_test_dtm))[:, 1]

	accuracy = accuracy_score(y_test, y_pred_classes)
	precision = precision_score(y_test, y_pred_classes)
	recall = recall_score(y_test, y_pred_classes)
	f1 = f1_score(y_test, y_pred_classes)
	auc = roc_auc_score(y_test, y_pred)
	matrix = confusion_matrix(y_test, y_pred_classes)

	print('Accuracy: %f' % accuracy)
	print('Precision: %f' % precision)
	print('Recall: %f' % recall)
	print('F1 score: %f' % f1)
	print('ROC AUC: %f' % auc)
	print(matrix)

	plot_roc(y_pred)

	return y_pred

## Loss train and val 

def plot_roc(y_pred):

	fpr, tpr, thresholds = roc_curve(y_test, y_pred)
	auc_logreg = auc(fpr, tpr)

	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr, tpr, label='Logistic Regression (area = {:.3f})'.format(auc_logreg))

	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('ROC curve')
	plt.savefig(os.path.join(work_dir, "Models/Logreg/roc.png"))
	plt.show()

y_pred = evaluation_plotting(logit)

############## XGBoost ####################

tfidf_transformer = TfidfTransformer()

tfidf_transformer.fit(x_train_dtm)
x_train_tfidf = tfidf_transformer.transform(x_train_dtm)

x_train_tfidf

pipe = Pipeline([
    ('bow', CountVectorizer()), 
    ('tfid', TfidfTransformer()),  
    ('model', xgb.XGBClassifier(
        learning_rate=0.1,
        max_depth=7,
        n_estimators=80,
        use_label_encoder=False,
        eval_metric='auc',
        # colsample_bytree=0.8,
        # subsample=0.7,
        # min_child_weight=5,
    ))
])

# Fit the pipeline with the data
pipe.fit(x_train, y_train)

y_pred_class = pipe.predict(x_test)
y_pred_train = pipe.predict(x_train)

print('Train: {}'.format(metrics.accuracy_score(y_train, y_pred_train)))
print('Test: {}'.format(metrics.accuracy_score(y_test, y_pred_class)))

conf_matrix(metrics.confusion_matrix(y_test, y_pred_class))