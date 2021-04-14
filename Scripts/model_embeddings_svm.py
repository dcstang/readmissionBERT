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
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, \
					confusion_matrix, precision_score, recall_score,  \
					f1_score, roc_curve, auc, classification_report, \
						plot_roc_curve

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

path = os.getcwd()
work_dir = os.path.join(path, 'Sem 2 - Machine Learning/Project')

dfUadm = pd.read_csv(os.path.join(work_dir, 'Data/lemmatized_stripped_CLEANED_FULL_UADM.csv'),
      dtype={'SUBJECT_ID' : 'UInt32', 'HADM_ID' : 'UInt32', 'TEXT_CONCAT' : 'string',
         'ADMISSION_TYPE' : 'string', 'ETHNICITY' : 'string', 'DIAGNOSIS' : 'string',
         'HOSPITAL_EXPIRE_FLAG' : 'bool', 'MORTALITY_30D' : 'bool', 'TARGET' : 'bool',
         'AGE' : 'UInt8'},
      parse_dates=['ADMITTIME', 'DISCHTIME', 'NEXT_UADMITTIME', 'DOD_SSN'],
      header=0)




######## TODO: WORD EMBEDDINGS ########
# 1. paddings first need to be done z/

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

def embed(corpus): 
    return word_tokenizer.texts_to_sequences(corpus)

# we just feed in the list of sentences, and we get the vector representation of each sentence
X_test = embed(msg_test)
X_test.shape


# we don't have enough memory to apply embeddings in one shot
# so we have to split the data into batches and concatenate them later
splits = np.array_split(msg_train, 5)
l = list()
for split in splits:
    l.append(embed(split))

X_train = tf.concat(l, axis=0)
del l
X_train.shape

# Calculate the length of our vocabulary
word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(train_tweets)

vocab_length = len(word_tokenizer.word_index) + 1
vocab_length

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


#################
class_weight = compute_class_weight(
    class_weight='balanced', classes=["Bullish","Bearish"], y=y_train
)
class_weight

X = dfUadm.TEXT_CONCAT
Y = dfUadm.TARGET

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.15, random_state=77, stratify=Y)

# SVM not scale invariant
sc_x = StandardScaler()
x_train_dtm = sc_x.fit_transform(x_train_dtm)
x_test_dtm = sc_x.transform(x_test_dtm)

# SMOTE on vectorized text
sm = SMOTE(sampling_strategy=0.5, random_state = 777)
x_train_dtm, y_train = sm.fit_resample(x_train_dtm, y_train)

print("Performing SMOTE on test_set:")
print("Number and prop(%) of cases   : ", (y_train == 1).sum(), 
			", % =", round((y_train == 1).sum()/len(y_train), 3))
print("Number and prop(%) of controls: ", (y_train == 0).sum(), 
			", % =", round((y_train == 0).sum()/len(y_train), 3))

print("x_train dims: ", x_train.shape)
print("x_test dims : ", x_test_dtm.shape, "\n")

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [10, 1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['linear', 'rbf', 'sigmoid']}  
  
grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3, n_jobs=-1) 
grid.fit(x_train_dtm, y_train) 

# print best parameter after tuning 
print(grid.best_params_) 


def use_svm(kernel, weights=None, penalty=1, gamma=1):
 
    print("\n", kernel, weights, penalty)
    clf = svm.SVC(kernel=kernel, class_weight=weights, C=penalty, gamma=gamma) 

    #Train the model using the training sets
    clf.fit(x_train_dtm, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(x_test_dtm)

    #Metrics
    fpr, tpr, _ = roc_curve(y_test, y_pred)

    ax = plt.gca()
    plot_roc_curve(clf, x_test_dtm, y_test, ax=ax, label=f'SVM {kernel} (AUC = {auc(fpr,tpr):.2f})')

    print(f'Confusion Matrix:{confusion_matrix(y_test, y_pred)}')
    print(f'F1 Score:{f1_score(y_test, y_pred)}')
    print(f'Accuracy Score: {accuracy_score(y_test, y_pred)}')
    print(f'Recall Score: {recall_score(y_test, y_pred)}')
    plt.savefig(os.path.join(work_dir, "Models/svm/countvec_roc.png"))
    plt.show()

    return clf, y_pred


best_svm, y_pred = use_svm(kernel='rbf', weights='balanced') # input best params 


# save model
with open(os.path.join(work_dir, "Models/svm/countvec_svm.pkl"), "wb") as file:
    pickle.dump(best_svm, file)

# save preds and y_test
np.savetext(os.path.join(work_dir, "Models/svm/y_pred.csv"), y_test, delimiter=',')
np.savetext(os.path.join(work_dir, "Models/svm/y_pred.csv"), y_pred, delimiter=',')




"""

def show_metrics(pred_tag, y_test):
    print("F1-score: ", f1_score(pred_tag, y_test))
    print("Precision: ", precision_score(pred_tag, y_test))
    print("Recall: ", recall_score(pred_tag, y_test))
    print("Acuracy: ", accuracy_score(pred_tag, y_test))
    print("-"*50)
    print(classification_report(pred_tag, y_test))
    

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



"""