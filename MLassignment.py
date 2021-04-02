import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tarfile
from io import BytesIO
import requests
from requests.exceptions import HTTPError
import xlsxwriter
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

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
            sub_df = df[['date_onset_symptoms','date_admission_hospital','date_confirmation','travel_history_dates','date_death_or_discharge']]
            print(sub_df)
            print(sub_df.iloc[[500000,1000000,1500000,2000000,2500000]])
            
            #print(df['chronic_disease_binary'].value_counts())
            #print("Number of patients with admission in hospital dates: ",2676311-df['date_admission_hospital'].isna().sum())
            #print("Outcome: ",df['outcome'].value_counts())
            #print("No outcome noted: ",df['outcome'].isna().sum())
            #print("Number of patients Date death or discharge noted: ",2676311-df['date_death_or_discharge'].isna().sum())
            #print(df.describe().to_string())
            #for v in df.columns:
            #    print(v,': ',2676311-df[v].isna().sum())
            
            #for v in df.columns:
            #    print(v,": ")
            #    print(df[v].value_counts())
            #    print("Nan: ",df[v].isna().sum())
            #    print()
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
            #print(df.keys())
            #Problem: Prediction of hospitalisation
            #print("Outcome",": ")
            #print(df['outcome'].value_counts())
            #print("Nan: ",df['outcome'].isna().sum())
            #print()


            #First clean the data: Drop rows,columns or impute missing data
            df['outcome'].replace(to_replace=['died','death','Dead','Death','Died','dead'],value='Deceased',inplace=True)
            df['outcome'].replace(to_replace=['recovered','released from quarantine','recovering at home 03.03.2020'],value='Recovered',inplace=True)
            df['outcome'].replace(to_replace=['discharged','Discharged from hospital','discharge'],value='Discharged',inplace=True)
            df['outcome'].replace(to_replace=['Under treatment','treated in an intensive care unit (14.02.2020)','critical condition, intubated as of 14.02.2020','Symptoms only improved with cough. Currently hospitalized for follow-up.','critical condition','severe illness','Critical condition','unstable','severe','Migrated','Migrated_Other'],value='Receiving Treatment',inplace=True)
            df['outcome'].replace(to_replace=['stable','stable condition'],value='Stable',inplace=True)
            
            
            #Work only with subset of data where outcome!=Nan
            #sub_df = df[df['outcome'].isna()==False]
            sub_df = df[df['outcome']!='https://www.mspbs.gov.py/covid-19.php']
            #Work with subset of data where sex and age is not missing
            smaller_df = sub_df[sub_df['sex'].isna()==False]
            smaller_df = smaller_df[smaller_df['age'].isna()==False]
            smaller_df = smaller_df[smaller_df['date_confirmation'].isna()==False]
            print(smaller_df)

            #Convert dates into dat format
            date_features=['date_onset_symptoms','date_admission_hospital','date_confirmation','travel_history_dates','date_death_or_discharge']
            for feature in date_features:
                smaller_df[feature]=pd.to_datetime(df[feature],format='%d.%m.%Y',errors='coerce')
            
            smaller_df['date_confirmation'+'_year']=smaller_df['date_confirmation'].dt.year 
            smaller_df['date_confirmation'+'_month']=smaller_df['date_confirmation'].dt.month
            smaller_df['date_confirmation'+'_day']=smaller_df['date_confirmation'].dt.day
            smaller_df['date_confirmation'+'_dayofweek']=smaller_df['date_confirmation'].dt.dayofweek
            print(smaller_df.info())

            ##Check which features to drop...
            smaller_df.drop('reported_market_exposure',axis=1,inplace=True)
            smaller_df.drop('sequence_available',axis=1,inplace=True)
            smaller_df.drop('notes_for_discussion',axis=1,inplace=True)
            smaller_df.drop('data_moderator_initials',axis=1,inplace=True)
            #Drop those with many nan values
            for v in smaller_df.columns:
                print(v,": ")
                print(smaller_df[v].value_counts())
                print("Nan: ",smaller_df[v].isna().sum())
                print()
            #Want to use dates, but there are a lot of missing dates in all the date features except for confirmation date
            #Make different calculations, some with and some without the dates
            #['date_onset_symptoms','date_admission_hospital','date_confirmation','travel_history_dates','date_death_or_discharge']
            smaller_df['difference_onset_admission']=(smaller_df['date_onset_symptoms'] - smaller_df['date_admission_hospital']).dt.days
            smaller_df['difference_onset_confirmation']=(smaller_df['date_onset_symptoms'] - smaller_df['date_confirmation']).dt.days
            smaller_df['difference_travel_onset']=(smaller_df['travel_history_dates'] - smaller_df['date_onset_symptoms']).dt.days
            smaller_df['difference_onset_deathordischarge']=(smaller_df['date_onset_symptoms'] - smaller_df['date_death_or_discharge']).dt.days
            smaller_df['difference_confirmation_admission']=(smaller_df['date_confirmation'] - smaller_df['date_admission_hospital']).dt.days
            smaller_df['difference_travel_admission']=(smaller_df['travel_history_dates'] - smaller_df['date_admission_hospital']).dt.days
            smaller_df['difference_admission_deathordischarge']=(smaller_df['date_admission_hospital'] - smaller_df['date_death_or_discharge']).dt.days
            smaller_df['difference_confirmation_deathordischarge']=(smaller_df['date_confirmation'] - smaller_df['date_death_or_discharge']).dt.days
            smaller_df['difference_travel_confirmation']=(smaller_df['travel_history_dates'] - smaller_df['date_confirmation']).dt.days


            ##Do something with different age stuff


            
            #Create a new column called deceased_binary, target label!!
            deceased_binary=[1 if (smaller_df['outcome'][i])=='Deceased' else 0 for i in smaller_df.index]
            #smaller_df['deceased_binary']=deceased_binary
            #Drop the outcome column, instead we now have the deceased binary
            smaller_df.drop('outcome',axis=1,inplace=True)
            y=pd.Series(deceased_binary)
            ##Split data into test and training set
            X_train, X_test, y_train, y_test = train_test_split(smaller_df, y, test_size=0.2,random_state=0,stratify=y,shuffle=True)
            print(X_train.shape, y_train.shape)
            print(X_test.shape, y_test.shape)

            '''
            #Example online
            # determine categorical and numerical features
            numerical_ix = X.select_dtypes(include=['int64', 'float64']).columns
            categorical_ix = X.select_dtypes(include=['object', 'bool']).columns
            # define the data preparation for the columns
            t = [('cat', OneHotEncoder(), categorical_ix), ('num', MinMaxScaler(), numerical_ix)]
            col_transform = ColumnTransformer(transformers=t)
            # define the model
            model = SVR(kernel='rbf',gamma='scale',C=100)
            # define the data preparation and modeling pipeline
            pipeline = Pipeline(steps=[('prep',col_transform), ('m', model)])
            # define the model cross-validation configuration
            cv = KFold(n_splits=10, shuffle=True, random_state=1)
            # evaluate the pipeline using cross validation and calculate MAE
            scores = cross_val_score(pipeline, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
            # convert MAE scores to positive values
            scores = absolute(scores)
            # summarize the model performance
            print('MAE: %.3f (%.3f)' % (mean(scores), std(scores)))
            
            ##impute data
            def plot_heatmap(df,title):
                fig, ax = plt.subplots()
                ax = sns.heatmap(df,cmap='BrBG',annot=True,annot_kws={'size':6})
                ax.set_title(title, fontsize=10, fontweight='bold')
                fig.tight_layout()
                plt.show()
            #K-nearest neighbour inputation?
            
            ##Encoding the data
            onehotencoder = OneHotEncoder(handle_unknown='ignore',sparse=False)
            X = onehotencoder.fit_transform(X).toarray()
            def encode_cyclic_data(df, feature, max_val): #sin,cos, transformation for cyclic data
                df[feature + '_sin'] = np.sin(2 * np.pi * df[feature]/max_val)
                df[feature + '_cos'] = np.cos(2 * np.pi * df[feature]/max_val)
                return df
            ##numeric features will be standardized(normalized)
            ##categorical data wil be onehotencoded

            ##Feature selection
            #X_new = SelectKBest(chi2, k=8).fit_transform(X_train, y)
            #print(X_new.shape)
            #print(X_new)
            '''
            




