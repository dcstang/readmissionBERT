import pandas as pd
import numpy as np
import os

path = os.getcwd()
work_dir = os.path.join(path, 'Sem 2 - Machine Learning/Project')

dfAdmPatient = pd.read_csv(os.path.join(work_dir, 'Data/cleanedadm.csv'),
      dtype={'SUBJECT_ID' : 'UInt32', 'HADM_ID' : 'UInt32',
         'ADMISSION_TYPE' : 'string', 'ETHNICITY' : 'string', 'DIAGNOSIS' : 'string',
         'HOSPITAL_EXPIRE_FLAG' : 'bool', 'MORTALITY_30D' : 'bool', 'TARGET' : 'bool'},
      parse_dates=['ADMITTIME', 'DISCHTIME', 'NEXT_UADMITTIME', 'DOD_SSN'],
      header=0)

dfConcNotes = pd.read_csv(os.path.join(work_dir, 'Data/cleanednotes.csv'),
      usecols = ['HADM_ID', 'TEXT_CONCAT'],
      dtype = {'HADM_ID' : 'UInt32', 'TEXT_CONCAT' : 'string'},
      header = 0)

# do merge
dfFull = pd.merge(dfAdmPatient, dfConcNotes, on='HADM_ID', how='left')
assert len(dfAdmPatient) == len(dfFull)

# drop columns without notes, len = 42,465
dfFull.drop(dfFull[pd.isna(dfFull.TEXT_CONCAT)].index, inplace=True)
dfFull.to_csv(os.path.join(work_dir, 'Data/merged.csv'), index=False)
