import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


path = os.getcwd()
work_dir = os.path.join(path, 'Sem 2 - Machine Learning/Project')


dfNotes = pd.read_csv(os.path.join(work_dir, 'Data/NOTEEVENTS.csv'),
    usecols=['SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'CATEGORY', 
        'DESCRIPTION', 'ISERROR', 'TEXT'],
    dtype={'SUBJECT_ID' : 'UInt32', 'HADM_ID' : 'UInt32', 'CATEGORY' : 'string', 
        'DESCRIPTION' : 'string', 'TEXT' : 'string'},
    parse_dates=['CHARTDATE'],
    na_values= '',
    header=0).fillna(False)

dfNotes['ISERROR'] = dfNotes['ISERROR'].astype('bool')