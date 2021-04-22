import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.metrics import roc_auc_score, accuracy_score, \
					confusion_matrix, precision_score, recall_score,  \
					f1_score, roc_curve, auc

path = os.getcwd()
work_dir = os.path.join(path, 'Sem 2 - Machine Learning/Project')


model_name = [
    'SWIFT Score                   ',
    'Count Vector + Logistic   ',
    'Embeddings + Logistic    ', 
    'Count Vector + SVM       ',
    'Embeddings + Deep LSTM   ']
model_paths = ['Models/SwiftScore', 'Models/Logreg',  'Models/Logreg', 'Models/svm', 'Models/LSTM']
model_nicknames = ['swiftscore', 'countvec_logreg', 'embedding_logreg', 'countvec_svm']

plt.close()
plt.style.use('seaborn')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 11.125))
fig.patch.set_facecolor('#EEEBE2')
ax.set_facecolor('#EEEBE2')

plt.plot([0, 1], [0, 1], color='#B2B2B2', linestyle=(0, (6, 25)))

for n in range(0, 4):
#for n in range(len(window_paths)):

    y_test = pd.read_csv(os.path.join(work_dir, model_paths[n], 
        '{}_y_test.csv'.format(model_nicknames[n])),
        header=None)
    y_pred = pd.read_csv(os.path.join(work_dir, model_paths[n], 
        '{}_y_pred.csv'.format(model_nicknames[n])),
        header=None)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc_keras = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'{model_name[n]} AUC: {auc_keras:.2f}')


plt.xticks(np.arange(0.0, 1.05, step=0.2))
plt.xlabel("False Positive Rate", fontsize=18)

plt.yticks(np.arange(0.0, 1.05, step=0.2))
plt.ylabel("True Positive Rate", fontsize=18)

ax.spines[['bottom', 'left']].set_visible(True)
ax.spines[['bottom', 'left']].set_color('black')

ax.grid(which='both', color='black')#F1EFE4

#plt.title('ROC Curve Analysis', fontweight='bold', fontsize=18)
plt.legend(prop={'size': 20}, loc='center right')

plt.show()
fig.tight_layout()
plt.savefig(os.path.join(work_dir, "Models/summary_roc.png"))