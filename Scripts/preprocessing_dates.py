import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


path = os.getcwd()
work_dir = os.path.join(path, 'Sem 2 - Machine Learning/Project')


dfAdm = pd.read_csv(os.path.join(work_dir, 'Data/ADMISSIONS.csv'),
    usecols=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME',
       'ADMISSION_TYPE', 'ETHNICITY', 'DIAGNOSIS'],
    dtype={'SUBJECT_ID' : 'UInt32', 'HADM_ID' : 'UInt32',
       'ADMISSION_TYPE' : 'string', 'ETHNICITY' : 'string', 'DIAGNOSIS' : 'string'},
    parse_dates=['ADMITTIME', 'DISCHTIME'],
    header=0)


# perform cleaning for readmissions
# make column for next admission
dfAdm = dfAdm.sort_values(['SUBJECT_ID', 'ADMITTIME']).reset_index(drop = True)
dfAdm['NEXT_UADMITTIME'] = dfAdm.groupby('SUBJECT_ID').ADMITTIME.shift(-1)

dfAdm['ADMISSION_TYPE'] = dfAdm['ADMISSION_TYPE'].astype(str)
dfAdm['NEXT_UADMISSION_TYPE'] = dfAdm.groupby('SUBJECT_ID').ADMISSION_TYPE.shift(-1)


# remove elective admissions
discardIdx = dfAdm[dfAdm['NEXT_UADMISSION_TYPE'] == 'ELECTIVE'].index
dfAdm.loc[discardIdx, 'NEXT_UADMITTIME'] = pd.NaT
dfAdm.loc[discardIdx, 'NEXT_UADMISSION_TYPE'] = np.NaN

dfAdm[['NEXT_UADMITTIME','NEXT_UADMISSION_TYPE']] = dfAdm.groupby(['SUBJECT_ID'])[['NEXT_UADMITTIME','NEXT_UADMISSION_TYPE']].fillna(method = 'bfill')

# get days to next admission
dfAdm['DAYS_NEXT_UADM']=  (dfAdm.NEXT_UADMITTIME - dfAdm.DISCHTIME).dt.total_seconds()/(24*60*60)

# drop no unplanned readmissions
# 47577 without readmissions, 11399 readmissions (6693 unique IDs)
dfAdm.drop(dfAdm[pd.isna(dfAdm.NEXT_UADMISSION_TYPE)].index, inplace=True)

dfAdm.to_csv(os.path.join(work_dir, 'Data/ureadmissions.csv'), index=False)