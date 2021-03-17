import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


path = os.getcwd()
work_dir = os.path.join(path, 'Sem 2 - Machine Learning/Project')


dfAdm = pd.read_csv(os.path.join(work_dir, 'Data/ADMISSIONS.csv'),
    usecols=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME',
       'ADMISSION_TYPE', 'ETHNICITY', 'DIAGNOSIS'],
    dtype={'SUBJECT_ID' : 'UInt32', 'HADM_ID' : 'UInt32',
       'ADMISSION_TYPE' : 'string', 'ETHNICITY' : 'string', 'DIAGNOSIS' : 'string'},
    parse_dates=['ADMITTIME', 'DISCHTIME'],
    header=0)

dfNotes = pd.read_csv(os.path.join(work_dir, 'Data/NOTEEVENTS.csv'),
    usecols=['SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'CATEGORY', 
        'DESCRIPTION', 'ISERROR', 'TEXT'],
    dtype={'SUBJECT_ID' : 'UInt32', 'HADM_ID' : 'UInt32', 'CATEGORY' : 'string', 
        'DESCRIPTION' : 'string', 'TEXT' : 'string'},
    parse_dates=['CHARTDATE'],
    na_values= '',
    header=0).fillna(False)

dfNotes['ISERROR'] = dfNotes['ISERROR'].astype('bool')

# perform cleaning for readmissions

# make column for next admission
dfAdm = dfAdm.sort_values(['SUBJECT_ID', 'ADMITTIME']).reset_index(drop = True)

# remove elective admissions