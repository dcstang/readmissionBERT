import os
import re
import pandas as pd
import numpy as np
import spacy
import scispacy
import eli5
import pickle 
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, \
					confusion_matrix, precision_score, recall_score,  \
					f1_score, roc_curve, auc, plot_roc_curve

import tensorflow as tf
import keras
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

path = os.getcwd()
work_dir = os.path.join(path, 'Sem 2 - Machine Learning/Project')

dfUadm = pd.read_csv(os.path.join(work_dir, 'Data/lemma_dfUadm.csv'),
      dtype={'SUBJECT_ID' : 'UInt32', 'HADM_ID' : 'UInt32', 'TEXT_CONCAT' : 'string',
         'ADMISSION_TYPE' : 'string', 'ETHNICITY' : 'string', 'DIAGNOSIS' : 'string',
         'HOSPITAL_EXPIRE_FLAG' : 'bool', 'MORTALITY_30D' : 'bool', 'TARGET' : 'bool',
         'AGE' : 'UInt8'},
      parse_dates=['ADMITTIME', 'DISCHTIME', 'NEXT_UADMITTIME', 'DOD_SSN'],
      header=0)

def final_clean(text):
    text = re.sub(r"[^a-zA-Z\d\s:]", "", text)
    text = re.sub(r"\b[a-zA-Z]\b", "", text)
    return text

dfUadm['TEXT_CONCAT'] = dfUadm['TEXT_CONCAT'].progress_apply(final_clean)

# train-test-split
X = dfUadm.TEXT_CONCAT
Y = dfUadm.TARGET

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.15, random_state=77, stratify=Y)

print("Initial stratified test_train_split 0.15 and info about test_set:")
print("Number and prop(%) of cases   : ", (y_train == 1).sum(), 
			", % =", round((y_train == 1).sum()/len(y_train), 3))
print("Number and prop(%) of controls: ", (y_train == 0).sum(), 
			", % =", round((y_train == 0).sum()/len(y_train), 3))

# embeddings for logreg
nlp = spacy.load("en_core_sci_md")

vocab_size = 100000
maxlen = 30

vectorizer = TextVectorization(
					max_tokens=vocab_size, 
					standardize=None,
					output_sequence_length=maxlen,
					output_mode='int')

vectorizer.adapt(x_train.to_numpy())

# get vocabulary ie. vector mappings of each word
vocab = vectorizer.get_vocabulary()
word_index = dict(zip(vocab, range(len(vocab))))

#generate the embedding matrix
embedding_dim = len(nlp('The').vector)
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for i, word in enumerate(vocab):
    embedding_matrix[i] = nlp(word).vector

print("Found %s word vectors." % len(embedding_matrix))


# load pre-trained embeddings into embedding layer
embedding_layer = Embedding(
    vocab_size,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False, 
	mask_zero=True
)

text_to_embeddings = tf.keras.Sequential([
	vectorizer,
    embedding_layer
])

x_train = text_to_embeddings.predict(x_train.to_numpy())
x_test = text_to_embeddings.predict(x_test.to_numpy())

# flatten to 2d np array
x_train = x_train.reshape(-1, x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(-1, x_test.shape[1]*x_test.shape[2])

# SMOTE on vectorized text
sm = SMOTE(sampling_strategy=0.6, random_state = 777)
x_train, y_train = sm.fit_resample(x_train, y_train)

print("Performing SMOTE on test_set:")
print("Number and prop(%) of cases   : ", (y_train == 1).sum(), 
			", % =", round((y_train == 1).sum()/len(y_train), 3))
print("Number and prop(%) of controls: ", (y_train == 0).sum(), 
			", % =", round((y_train == 0).sum()/len(y_train), 3))

print("x_train dims: ", x_train.shape)
print("x_test dims : ", x_test.shape, "\n")

# implement gridsearchCV
grid = {"C": np.logspace(-3, 3, 7), "penalty" : ["l1", "l2"]}

logreg=LogisticRegression(max_iter=300, solver='saga', n_jobs=20)
logreg_cv = GridSearchCV(logreg, grid, cv=5)
best_log = logreg_cv.fit(x_train, y_train)

print("tuned hyperparameters :(best parameters) ",best_log.best_params_)
print("accuracy :",best_log.best_score_)

print('Best Penalty:', best_log.best_estimator_.get_params()['penalty'])
print('Best C:', best_log.best_estimator_.get_params()['C'])

# do full analysis using best hyperparameters
logit = LogisticRegression(C=best_log.best_estimator_.get_params()['C'], 
							solver='saga', multi_class='auto', 
                            n_jobs=20, max_iter=300, 
							penalty=best_log.best_estimator_.get_params()['penalty'])

logit.fit(x_train, y_train)


def evaluation_plotting(model):

	y_pred_classes = model.predict(x_test)
	y_pred = (model.predict_proba(x_test))[:, 1]

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
	plt.plot(fpr, tpr, label='Embedding + Logistic Regression (area = {:.3f})'.format(auc_logreg))

	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('ROC curve')
	plt.savefig(os.path.join(work_dir, "Models/Logreg/embedding_logreg_roc.png"))
	plt.show()

y_pred = evaluation_plotting(logit)

# save model
with open(os.path.join(work_dir, "Models/Logreg/embedding_logreg.pkl"), "wb") as file:
    pickle.dump(logit, file)

# save preds and y_test
np.savetxt(os.path.join(work_dir, "Models/Logreg/embedding_y_test.csv"), y_test, delimiter=',')
np.savetxt(os.path.join(work_dir, "Models/Logreg/embedding_y_pred.csv"), y_pred, delimiter=',')