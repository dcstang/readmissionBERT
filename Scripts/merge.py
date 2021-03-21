import pandas as pd
import numpy as np
import os

path = os.getcwd()
work_dir = os.path.join(path, 'Sem 2 - Machine Learning/Project')

dfAdm = pd.read_csv(os.path.join(work_dir, 'Data/cleanedadm.csv'),
    dtype={'SUBJECT_ID' : 'UInt32', 'HADM_ID' : 'UInt32',
       'ADMISSION_TYPE' : 'string', 'ETHNICITY' : 'string', 'DIAGNOSIS' : 'string'},
    parse_dates=['ADMITTIME', 'DISCHTIME', 'NEXT_UADMITTIME'],
    header=0)
