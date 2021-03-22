import pandas as pd
import numpy as np
import os
import datetime

path = os.getcwd()
work_dir = os.path.join(path, 'Sem 2 - Machine Learning/Project')

dfAdm = pd.read_csv(os.path.join(work_dir, 'Data/ADMISSIONS.csv'),
      usecols=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME',
         'ADMISSION_TYPE', 'ETHNICITY', 'DIAGNOSIS'],
      dtype={'SUBJECT_ID' : 'UInt32', 'HADM_ID' : 'UInt32',
         'ADMISSION_TYPE' : 'string', 'ETHNICITY' : 'string', 'DIAGNOSIS' : 'string'},
      parse_dates=['ADMITTIME', 'DISCHTIME'],
      header=0)

dfPatient = pd.read_csv(os.path.join(work_dir, 'Data/PATIENTS.csv'),
      usecols=['SUBJECT_ID', 'GENDER', 'DOD_SSN'],
      dtype={'SUBJECT_ID' : 'UInt32', 'GENDER' : 'string'},
      parse_dates=['DOD_SSN'],
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

# cleaned unplanned readmissions total
# but drop newborn = 51113 x 10
dfAdm.drop(dfAdm[dfAdm.ADMISSION_TYPE == 'NEWBORN'].index, inplace=True)
dfAdm.to_csv(os.path.join(work_dir, 'Data/cleanedadm.csv'), index=False)

# get date of deaths dod_ssn (PATIENTS.CSV)
# make sure death doesn't occur in next 30 days 
dfAdm = pd.read_csv(os.path.join(work_dir, 'Data/cleanedadm.csv'),
      dtype={'SUBJECT_ID' : 'UInt32', 'HADM_ID' : 'UInt32',
         'ADMISSION_TYPE' : 'string', 'ETHNICITY' : 'string', 'DIAGNOSIS' : 'string'},
      parse_dates=['ADMITTIME', 'DISCHTIME', 'NEXT_UADMITTIME'],
      header=0)

dfAdmPatient = pd.merge(dfAdm, dfPatient, on=['SUBJECT_ID'], how='left')
assert len(dfAdm) == len(dfAdmPatient)

# forgot mortality cases, pull in and remove these 
died = pd.read_csv(os.path.join(work_dir, 'Data/ADMISSIONS.csv'),
      usecols=['HADM_ID', 'HOSPITAL_EXPIRE_FLAG'],
      dtype={'HADM_ID' : 'UInt32', 'HOSPITAL_EXPIRE_FLAG' : 'bool'},
      header=0)

dfAdmPatient = pd.merge(dfAdmPatient, died, on=['HADM_ID'], how='left')
assert len(dfAdm) == len(dfAdmPatient) # len = 51,113
dfAdmPatient.drop(dfAdmPatient[dfAdmPatient.HOSPITAL_EXPIRE_FLAG == True].index, inplace=True)
# result len = 45321

# make outcome / target variable
# 0 = no readmission, survived 30-d post discharge
# 1 = readmission to ICU in 30-days 
dfAdmPatient['MORTALITY_30D'] = ((dfAdmPatient.DOD_SSN - dfAdmPatient.DISCHTIME) < datetime.timedelta(days=31))
dfAdmPatient['TARGET'] = (dfAdmPatient.DAYS_NEXT_UADM < 31)

# drop those who were discharged, and not readmitted because terminal
dfAdmPatient.drop(dfAdmPatient[(dfAdmPatient.TARGET == False) & (dfAdmPatient.MORTALITY_30D == True)].index,
      inplace=True)

# save output
# 3028 (true) readmissions / 40649 with false label
dfAdmPatient.to_csv(os.path.join(work_dir, 'Data/cleanedadm.csv'), index=False)