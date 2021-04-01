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
            print(df)
            print(df.info())
            #sub_df = df[['date_onset_symptoms','date_admission_hospital','date_confirmation','travel_history_dates','date_death_or_discharge']]
            #print(sub_df)
            #print(sub_df.iloc[[500000,1000000,1500000,2000000,2500000]])
            #print(df['chronic_disease_binary'].value_counts())
            #print("Number of patients with admission in hospital dates: ",2676311-df['date_admission_hospital'].isna().sum())
            #print("Outcome: ",df['outcome'].value_counts())
            #print("No outcome noted: ",df['outcome'].isna().sum())
            #print("Number of patients Date death or discharge noted: ",2676311-df['date_death_or_discharge'].isna().sum())
            print(df.describe().to_string())
            #for v in df.columns:
            #    print(v,': ',2676311-df[v].isna().sum())
            for v in df.columns:
                print(v,": ")
                print(df[v].value_counts())
                print("Nan: ",df[v].isna().sum())
                print()
            #df_1 = df.iloc[:100]
            #df_2 = df.iloc[1000001:1000100]
            #df_3 = df.iloc[2500000:]
            #writer = pd.ExcelWriter('./data.xlsx',engine='xlsxwriter')
            #df_1.to_excel(writer, sheet_name='Sheet1')
            #df_2.to_excel(writer,sheet_name='Sheet2')
            #df_3.to_excel(writer,sheet_name='Sheet3')
            #writer.save()
            df.to_csv('./covid_data.csv')

##Work with local copy of data for now
#Problem: Prediction of hospitalisation
#First work with only subset of data where outcome!=Nan

