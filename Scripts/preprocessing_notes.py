import pandas as pd
import numpy as np
import os

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

# drop notes which are erroneous, n = 886
dfNotes.drop(dfNotes[dfNotes.ISERROR == True].index, inplace=True)

# keep only 'Discharge summary', 'Nursing', 'Consult' and 'General'
# n remainding = 291,168
dfNotes = dfNotes[dfNotes.CATEGORY.isin(['Discharge summary', 'General', 'Consult', 'Nursing'])]
dfNotes = dfNotes.sort_values(['SUBJECT_ID','CHARTDATE', 'CATEGORY'], 
            ascending=[True, True, False]).reset_index(drop = True)

# flatten notes
dfConcNotes = dfNotes[['SUBJECT_ID', 'TEXT']].copy()
dfConcNotes = (
    dfConcNotes
    .groupby('SUBJECT_ID')['TEXT']
    .agg(' '.join)
    .reset_index()
    .rename(columns={"TEXT":"TEXT_CONCAT"})
)

# checkpoint
dfConcNotes.to_csv(os.path.join(work_dir, 'Data/cleanednotes.csv'), index=True)