import pandas as pd
import numpy as np
import scipy
import os
import re
import eli5
import pickle 
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, \
					confusion_matrix, precision_score, recall_score,  \
					f1_score, roc_curve, auc, classification_report, \
				plot_roc_curve


path = os.getcwd()
work_dir = os.path.join(path, 'Sem 2 - Machine Learning/Project')

dfUadm = pd.read_csv(os.path.join(work_dir, 'Data/lemmatized_stripped_CLEANED_FULL_UADM.csv'),
      dtype={'SUBJECT_ID' : 'UInt32', 'HADM_ID' : 'UInt32', 'TEXT_CONCAT' : 'string',
         'ADMISSION_TYPE' : 'string', 'ETHNICITY' : 'string', 'DIAGNOSIS' : 'string',
         'HOSPITAL_EXPIRE_FLAG' : 'bool', 'MORTALITY_30D' : 'bool', 'TARGET' : 'bool',
         'AGE' : 'UInt8'},
      parse_dates=['ADMITTIME', 'DISCHTIME', 'NEXT_UADMITTIME', 'DOD_SSN'],
      header=0)


# load test_train split countvector data
# train-test-split
X = dfUadm.TEXT_CONCAT
Y = dfUadm.TARGET

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.15, random_state=77, stratify=Y)

# load sparse matrix checkpoint
x_train_dtm = scipy.sparse.load_npz(os.path.join(work_dir, 'Data/Models/CountVector/x_train.npz'))
x_test_dtm = scipy.sparse.load_npz(os.path.join(work_dir, 'Data/Models/CountVector/x_test.npz'))

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
np.savetxt(os.path.join(work_dir, "Models/svm/y_pred.csv"), y_test, delimiter=',')
np.savetxt(os.path.join(work_dir, "Models/svm/y_pred.csv"), y_pred, delimiter=',')