import pandas as pd
import numpy as np
import os
import datetime
import re
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, \
					confusion_matrix, precision_score, recall_score,  \
					f1_score, roc_curve, auc



path = os.getcwd()
work_dir = os.path.join(path, 'Sem 2 - Machine Learning/Project')

dfUadm = pd.read_csv(os.path.join(work_dir, 'Data/lemma_dfUadm.csv'),
        usecols=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'TARGET'],
        dtype={'SUBJECT_ID' : 'UInt32', 'HADM_ID' : 'UInt32', 'TARGET' : 'bool'},
        parse_dates=['ADMITTIME', 'DISCHTIME'],
        header=0)

dfUadm['ADM_DURATION'] = dfUadm['DISCHTIME'] - dfUadm['ADMITTIME']
dfUadm['ADM_DURATION'] = dfUadm['ADM_DURATION'].dt.days

itemSought = set([190, 198, 778, 779, 2981, 227010, 220739, 223900, 223901, 227038])
ids = set(dfUadm['HADM_ID'].unique())

dfCharts = pd.read_csv(os.path.join(work_dir, 'Data/CHARTEVENTS.csv'),
        usecols=['HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUE'],
        dtype={'HADM_ID': 'UInt32', 'ITEMID': 'UInt32', 'VALUE' : 'O'},
        parse_dates=['CHARTTIME'],
        header=0, chunksize=500000)

smallcharts = pd.DataFrame()

n = 1
for chunk in dfCharts:
    print(chunk.shape)
    print(n)
    chunk = chunk[(chunk['HADM_ID'].isin(ids)) & (chunk['ITEMID'].isin(itemSought)) ]
    smallcharts = smallcharts.append(chunk)
    print(smallcharts)
    n += 1

smallcharts.to_csv(os.path.join(work_dir, 'Data/smallcharts.csv'), index=False)
smallcharts = pd.read_csv(os.path.join(work_dir, 'Data/smallcharts.csv'), 
        dtype={'HADM_ID': 'UInt32', 'ITEMID': 'UInt32', 'VALUE' : 'O'},
        parse_dates=['CHARTTIME'], header=0)

smallcharts['VALUE'] = smallcharts['VALUE'].replace(['To Speech', 'Spontaneously', 
        'No Response-ETT', 'Localizes Pain', 'Oriented', 'Obeys Commands', 
        'None', 'Flex-withdraws', 'To Pain', 'No response', 'No Response',
        'Confused', 'Abnormal Flexion', 'Abnormal extension', 'Incomprehensible sounds',
        'Inappropriate Words'], [3, 4, 1, 5, 5, 6, 1, 3, 2, 1, 1, 4, 3, 2, 2, 3])

smallcharts['VALUE'] = pd.to_numeric(smallcharts['VALUE'], downcast='float')
smallcharts = smallcharts.sort_values(by=['CHARTTIME']).drop_duplicates(subset=['HADM_ID', 'ITEMID'], keep='last')
smallcharts = smallcharts.pivot(index='HADM_ID', columns='ITEMID', values='VALUE')

dfloc = pd.read_csv(os.path.join(work_dir, 'Data/ADMISSIONS.csv'),
        usecols=['HADM_ID', 'ADMISSION_LOCATION'],
        dtype={'HADM_ID': 'UInt32', 'ADMISSION_LOCATION': 'string'},
        header=0)

dfloc['ADMISSION_LOCATION'] = dfloc['ADMISSION_LOCATION'].replace('EMERGENCY ROOM ADMIT', 0)
dfloc['ADMISSION_LOCATION'] = dfloc['ADMISSION_LOCATION'].replace(regex=r'[a-zA-Z]', value=8)
dfloc['ADMISSION_LOCATION'] = pd.to_numeric(dfloc['ADMISSION_LOCATION'], downcast='unsigned')

df_Swift = pd.merge(dfUadm, dfloc, how='left', on='HADM_ID')
df_Swift = pd.merge(df_Swift, smallcharts, how='left', on='HADM_ID')
assert len(df_Swift) == len(dfUadm)

# get last GCS (198 and 220739, 223900, 223901)
# get last PaO2 (779) and Fio2 (190/2981 and 227010) >> ratio 
# get last PaCO2 (778 and 227038)

# summarize GCS into one score
df_Swift[198] = df_Swift.apply(
    lambda row: row[220739]+row[223900]+row[223901] if np.isnan(row[198]) else row[198],
    axis = 1
)
df_Swift.drop(columns=[220739, 223900, 223901], inplace=True)

# calculate ratio of inspired oxygen
df_Swift['O2RATIO'] = df_Swift[779]/df_Swift[190]
df_Swift.drop(columns=[779, 190], inplace=True)
df_Swift.rename(columns={198: 'GCS', 778: 'PACO2'}, inplace=True)

#### calculate SWIFT score
# ICU stay scoring
conditions = [(df_Swift['ADM_DURATION'] < 2), 
    (df_Swift['ADM_DURATION'] >= 2) & (df_Swift['ADM_DURATION'] <= 10), 
    (df_Swift['ADM_DURATION'] > 10)]
admDurationScore = [0, 1, 14]
df_Swift['ADMDURSCORE'] = np.select(conditions, admDurationScore, default=0)

# pao2.fio2 ratio score
conditions = [(df_Swift['O2RATIO'] < 100), 
    (df_Swift['O2RATIO'] >= 100) & (df_Swift['O2RATIO'] < 150), 
    (df_Swift['O2RATIO'] >= 150) & (df_Swift['O2RATIO'] < 400), 
    (df_Swift['O2RATIO'] >= 400)]
inspiredScore = [13, 10, 5, 0]
df_Swift['INSPSCORE'] = np.select(conditions, inspiredScore, default=0)

# GCS score
conditions = [(df_Swift['GCS'] < 8), 
    (df_Swift['GCS'] >= 8) & (df_Swift['GCS'] < 11), 
    (df_Swift['GCS'] >= 11) & (df_Swift['GCS'] < 14), 
    (df_Swift['GCS'] >= 14)]
comaScore = [24, 14, 6, 0]
df_Swift['GCSCORE'] = np.select(conditions, comaScore, default=0)

# PaCO2 score
conditions = [(df_Swift['PACO2'] > 45), (df_Swift['PACO2'] <= 45)]
co2Score = [5, 0]
df_Swift['PACO2SCORE'] = np.select(conditions, co2Score, default=0)


df_Swift['SWIFTSCORE'] = df_Swift['ADMDURSCORE'] + df_Swift['INSPSCORE'] + \
    df_Swift['GCSCORE'] + df_Swift['PACO2SCORE']


# make class preds
conditions = [(df_Swift['SWIFTSCORE'] >= 15), (df_Swift['SWIFTSCORE'] < 15)]
swiftScore = [True, False]
df_Swift['SWIFTPRED'] = np.select(conditions, swiftScore, default=False)

# make prob preds ~ 3.5% per point on SWIFT (simplified implementation)
# https://doi.org/10.1371/journal.pone.0143127
# ranges from 1 to 48

df_Swift['SWIFTPROB'] = df_Swift['SWIFTSCORE'] * 0.035

df_Swift.to_csv(os.path.join(work_dir, 'Data/SWIFTSCORE.csv'), index=False)

# calculate ROC
accuracy = accuracy_score(df_Swift['TARGET'], df_Swift['SWIFTPRED'])
precision = precision_score(df_Swift['TARGET'], df_Swift['SWIFTPRED'])
recall = recall_score(df_Swift['TARGET'], df_Swift['SWIFTPRED'])
f1 = f1_score(df_Swift['TARGET'], df_Swift['SWIFTPRED'])
auc = roc_auc_score(df_Swift['TARGET'], df_Swift['SWIFTPROB'])
matrix = confusion_matrix(df_Swift['TARGET'], df_Swift['SWIFTPRED'])

print('Accuracy: %f' % accuracy)
print('Precision: %f' % precision)
print('Recall: %f' % recall)
print('F1 score: %f' % f1)
print('ROC AUC: %f' % auc)
print(matrix)

df_Swift['SWIFTPROB'].to_csv(os.path.join(work_dir, 'Models/SwiftScore/y_pred.csv'), index=False, header=False)
df_Swift['TARGET'].to_csv(os.path.join(work_dir, 'Models/SwiftScore/y_test.csv'), index=False, header=False)