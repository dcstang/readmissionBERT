import os
import re
import pandas as pd
import numpy as np
import scipy
import spacy
import scispacy
import eli5
import pickle 
from tqdm import tqdm
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, \
					confusion_matrix, precision_score, recall_score,  \
					f1_score, roc_curve, auc


path = os.getcwd()
work_dir = os.path.join(path, 'Sem 2 - Machine Learning/Project')

## TODO: change path to remerged post lemmatization
dfUadm = pd.read_csv(os.path.join(work_dir, 'Data/CLEANED_FULL_UADM.csv'),
      dtype={'SUBJECT_ID' : 'UInt32', 'HADM_ID' : 'UInt32', 'TEXT_CONCAT' : 'string',
         'ADMISSION_TYPE' : 'string', 'ETHNICITY' : 'string', 'DIAGNOSIS' : 'string',
         'HOSPITAL_EXPIRE_FLAG' : 'bool', 'MORTALITY_30D' : 'bool', 'TARGET' : 'bool',
         'AGE' : 'UInt8'},
      parse_dates=['ADMITTIME', 'DISCHTIME', 'NEXT_UADMITTIME', 'DOD_SSN'],
      header=0)

tqdm.pandas(desc="Pandas Apply Progress")


# train-test-split
X = dfUadm.TEXT_CONCAT
Y = dfUadm.TARGET

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.15, random_state=77, stratify=Y)

print("Initial stratified test_train_split 0.15 and info about test_set:")
print("Number and prop(%) of cases   : ", (y_train == 1).sum(), 
			", % =", round((y_train == 1).sum()/len(y_train), 3))
print("Number and prop(%) of controls: ", (y_train == 0).sum(), 
			", % =", round((y_train == 0).sum()/len(y_train), 3))

# instantiate the vectorizer
vect_tunned = CountVectorizer(ngram_range=(1,2), lowercase=False, min_df=0.1, max_df=0.85, max_features=10000)
vect_tunned.fit(x_train)

with open(os.path.join(work_dir, 'Data/Models/CountVector/CountVect'), 'wb') as fout:
    pickle.dump(vect_tunned, fout)

# create a document-term matrix from train and test sets
x_train_dtm = vect_tunned.transform(x_train)
x_test_dtm = vect_tunned.transform(x_test)

"""
# checkpoint save sparse matrix
scipy.sparse.save_npz(os.path.join(work_dir, 'Data/Models/CountVector/x_train.npz'), x_train_dtm)
scipy.sparse.save_npz(os.path.join(work_dir, 'Data/Models/CountVector/x_test.npz'), x_test_dtm)
"""

# load sparse matrix checkpoint
x_train_dtm = scipy.sparse.load_npz(os.path.join(work_dir, 'Data/Models/CountVector/x_train.npz'))
x_test_dtm = scipy.sparse.load_npz(os.path.join(work_dir, 'Data/Models/CountVector/x_test.npz'))

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

# implement gridsearchCV
grid = {"C": np.logspace(-3, 3, 7), "penalty" : ["l1", "l2"]}

logreg=LogisticRegression(max_iter=300)
logreg_cv = GridSearchCV(logreg, grid, CV=5)
logreg_cv.fit(x_train_dtm, y_train)

print("tuned hyperparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)

# do full analysis using best hyperparameters
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
	plt.savefig(os.path.join(work_dir, "Models/Logreg/countvec_roc.png"))
	plt.show()

y_pred = evaluation_plotting(logit)

# save model
with open(os.path.join(work_dir, "Models/Logreg/countvec_logreg.pkl"), "wb") as file:
    pickle.dump(logit, file)

# save preds and y_test
np.savetext(os.path.join(work_dir, "Models/svm/y_pred.csv"), y_test, delimiter=',')
np.savetext(os.path.join(work_dir, "Models/svm/y_pred.csv"), y_pred, delimiter=',')