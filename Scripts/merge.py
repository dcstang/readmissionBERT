import pandas as pd
import numpy as np
import os
import spacy
import scispacy

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

# pull in age from NOTEVENTS as function of note - date of birth (from PATIENTS)
# do this based on HADM_ID

dfChartDate = pd.read_csv(os.path.join(work_dir, 'Data/NOTEEVENTS.csv'),
                        usecols = ['SUBJECT_ID', 'HADM_ID', 'CHARTDATE'],
                        dtype = {'SUBJECT_ID' : 'UInt32', 'HADM_ID' : 'UInt32'},                        
                        parse_dates = ['CHARTDATE'])

dfDob = pd.read_csv(os.path.join(work_dir, 'Data/PATIENTS.csv'),
                        usecols = ['SUBJECT_ID', 'DOB'],
                        dtype = {'SUBJECT_ID' : 'UInt32'},
                        parse_dates = ['DOB'])

dfAge = pd.merge(dfChartDate, dfDob, on='SUBJECT_ID', how='left')
assert len(dfAge) == len(dfChartDate)
s
dfAge['AGE'] = dfAge['CHARTDATE'].dt.year - dfAge['DOB'].dt.year
dfAge.loc[dfAge['AGE'] > 200, 'AGE'] = 90

dfAge.to_csv(os.path.join(work_dir, 'Data/dob.csv'), index=False)
dfAge = pd.read_csv(os.path.join(work_dir, 'Data/dob.csv'),
      dtype={'SUBJECT_ID' : 'UInt32', 'HADM_ID' : 'UInt32', 'AGE' : 'UInt8'},
      parse_dates=['CHARTDATE', 'DOB'],
      header=0)
dfAge.drop(columns=['SUBJECT_ID', 'CHARTDATE', 'DOB'], inplace=True)
dfAge = dfAge.drop_duplicates(subset="HADM_ID")

# merge back into dfFull
dfFull = pd.read_csv(os.path.join(work_dir, 'Data/merged.csv'),
      dtype={'SUBJECT_ID' : 'UInt32', 'HADM_ID' : 'UInt32', 'TEXT_CONCAT' : 'string',
         'ADMISSION_TYPE' : 'string', 'ETHNICITY' : 'string', 'DIAGNOSIS' : 'string',
         'HOSPITAL_EXPIRE_FLAG' : 'bool', 'MORTALITY_30D' : 'bool', 'TARGET' : 'bool'},
      parse_dates=['ADMITTIME', 'DISCHTIME', 'NEXT_UADMITTIME', 'DOD_SSN'],
      header=0)

dfFull = pd.merge(dfFull, dfAge, on='HADM_ID', how='left')

# do a bit of cleaning on ETHNICITY and DIAGNOSIS

asian = dict.fromkeys(['ASIAN','ASIAN - ASIAN INDIAN','ASIAN - CAMBODIAN',
      'ASIAN - CHINESE','ASIAN - FILIPINO', 'ASIAN - JAPANESE', 'ASIAN - KOREAN',
      'ASIAN - OTHER', 'ASIAN - THAI', 'ASIAN - VIETNAMESE', 'MIDDLE EASTERN'], 'asian')    

white = dict.fromkeys([ 'WHITE', 'WHITE - BRAZILIAN', 'WHITE - EASTERN EUROPEAN', 
      'WHITE - OTHER EUROPEAN', 'WHITE - RUSSIAN'], 'white')

black = dict.fromkeys([ 'BLACK/AFRICAN', 'BLACK/AFRICAN AMERICAN',
       'BLACK/CAPE VERDEAN','BLACK/HAITIAN'], 'black')

hispanic = dict.fromkeys([ 'HISPANIC OR LATINO', 'HISPANIC/LATINO - CENTRAL AMERICAN (OTHER)',
       'HISPANIC/LATINO - COLOMBIAN', 'HISPANIC/LATINO - CUBAN', 'HISPANIC/LATINO - DOMINICAN',
       'HISPANIC/LATINO - GUATEMALAN', 'HISPANIC/LATINO - HONDURAN', 'HISPANIC/LATINO - MEXICAN',
       'HISPANIC/LATINO - PUERTO RICAN', 'HISPANIC/LATINO - SALVADORAN', 'PORTUGUESE',
       'SOUTH AMERICAN'], 'hispanic')

dfFull = dfFull.replace(asian)
dfFull = dfFull.replace(white)
dfFull = dfFull.replace(black)
dfFull = dfFull.replace(hispanic)

allowed_vals = ['asian', 'black', 'white', 'hispanic']
dfFull.loc[~dfFull['ETHNICITY'].isin(allowed_vals), 'ETHNICITY'] = "other"

# diagnosis cleaning

nlp = spacy.load('en_core_sci_scibert', exclude=['ner'])

def cleaned_diagnosis(text):
    
    text = text.lower()
    tokens = nlp.tokenizer(text)
    tokenised_text = ""
    
    for token in tokens:
        tokenised_text = tokenised_text + str(token) + " "
    
    tokenised_text = ' '.join(tokenised_text.split())
    
    return tokenised_text 


dfFull.at[20785, 'DIAGNOSIS'] = 'NON SMALL CELL LUNG CARCINOMA'
dfFull["DIAGNOSIS"] = dfFull["DIAGNOSIS"].apply(cleaned_diagnosis)
dfFull.head()

def manual_dx_cleaning(text):

    text = re.sub(r"\/sda", "", text)
    text = re.sub(r"\/pna", "", text)

    return text

dfFull['DIAGNOSIS'] = dfFull['DIAGNOSIS'].apply(manual_dx_cleaning)

dfFull.to_csv(os.path.join(work_dir, 'Data/merged.csv'), index=False)