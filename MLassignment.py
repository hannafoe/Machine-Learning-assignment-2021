import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import tarfile
from io import BytesIO
import requests
from requests.exceptions import HTTPError
import xlsxwriter

#URL for data 
url = 'https://github.com/beoutbreakprepared/nCoV2019/blob/master/latest_data/latestdata.tar.gz?raw=true'
try:
    res = requests.get(url,stream=True,timeout=1)
    # If the response was successful, no Exception will be raised
    res.raise_for_status()
except HTTPError as http_err:
    print(f'HTTP error occurred: {http_err}')
except Exception as err:
    print(f'Other error occurred: {err}')
else:
    print('Success!')
    #get data & import  as a dataframe
    with tarfile.open(fileobj=BytesIO(res.raw.read()),mode='r:gz') as tar:
        for m in tar.getmembers():
            file = tar.extractfile(m)
            df = pd.read_csv(file)
            #print(df)
            print(df.info())
            #sub_df = df[['date_onset_symptoms','date_admission_hospital','date_confirmation','travel_history_dates','date_death_or_discharge']]
            #print(sub_df)
            #print(sub_df.iloc[[500000,1000000,1500000,2000000,2500000]])
            #print(df['chronic_disease_binary'].value_counts())
            #print("Number of patients with admission in hospital dates: ",2676311-df['date_admission_hospital'].isna().sum())
            #print("Outcome: ",df['outcome'].value_counts())
            #print("No outcome noted: ",df['outcome'].isna().sum())
            #print("Number of patients Date death or discharge noted: ",2676311-df['date_death_or_discharge'].isna().sum())
            #print(df.describe().to_string())
            #for v in df.columns:
            #    print(v,': ',2676311-df[v].isna().sum())
            '''
            for v in df.columns:
                print(v,": ")
                print(df[v].value_counts())
                print("Nan: ",df[v].isna().sum())
                print()'''
            #df_1 = df.iloc[:100]
            #df_2 = df.iloc[1000001:1000100]
            #df_3 = df.iloc[2500000:]
            #writer = pd.ExcelWriter('./data.xlsx',engine='xlsxwriter')
            #df_1.to_excel(writer, sheet_name='Sheet1')
            #df_2.to_excel(writer,sheet_name='Sheet2')
            #df_3.to_excel(writer,sheet_name='Sheet3')
            #writer.save()
            #df.to_csv('./covid_data.csv')

            #Look at data
            print(df.keys())
            #Problem: Prediction of hospitalisation
            print("Outcome",": ")
            print(df['outcome'].value_counts())
            print("Nan: ",df['outcome'].isna().sum())
            print()
            #First clean the data
            df['outcome'].replace(to_replace=['died','death','Dead','Death','Died','dead'],value='Deceased',inplace=True)
            df['outcome'].replace(to_replace=['recovered','released from quarantine','recovering at home 03.03.2020'],value='Recovered',inplace=True)
            df['outcome'].replace(to_replace=['discharged','Discharged from hospital','discharge'],value='Discharged',inplace=True)
            df['outcome'].replace(to_replace=['Under treatment','treated in an intensive care unit (14.02.2020)','critical condition, intubated as of 14.02.2020','Symptoms only improved with cough. Currently hospitalized for follow-up.','critical condition','severe illness','Critical condition','unstable','severe','Migrated','Migrated_Other'],value='Receiving Treatment',inplace=True)
            df['outcome'].replace(to_replace=['stable','stable condition'],value='Stable',inplace=True)
            
            
            #First work with only subset of data where outcome!=Nan
            sub_df = df[df['outcome'].isna()==False]
            sub_df = sub_df[sub_df['outcome']!='https://www.mspbs.gov.py/covid-19.php']
            print("Outcome: ")
            print(sub_df['outcome'].value_counts())
            print("Nan: ",sub_df['outcome'].isna().sum())

            smaller_df = sub_df[sub_df['sex'].isna()==False]
            smaller_df = smaller_df[smaller_df['age'].isna()==False]
            print(smaller_df)

            #Work only with data, where sex, age and outcome is labelled
            #Split data into test and training set
            #Create a new column called deceased_binary
            y=[True if (smaller_df['outcome'][i])=='Deceased' else False for i in range(len(smaller_df['outcome'])) ]
            smaller_df['deceased_binary']=y
            X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
            print(X_train.shape, y_train.shape)
            print(X_test.shape, y_test.shape)

