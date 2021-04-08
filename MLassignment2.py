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
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer,IterativeImputer
from sklearn.metrics import precision_recall_curve,classification_report,roc_curve, auc
from sklearn.metrics import average_precision_score
from sklearn.inspection import permutation_importance
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import re
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier,NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from statistics import mean
import eli5
from eli5.sklearn import PermutationImportance
import scipy.stats as ss


url = 'https://github.com/beoutbreakprepared/nCoV2019/blob/master/latest_data/latestdata.tar.gz?raw=true'
try:
    res = requests.get(url,stream=True,timeout=5)
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
            def plot_heatmap(df,title):
                fig, ax = plt.subplots()
                ax = sns.heatmap(df,cmap='BrBG',annot=True,annot_kws={'size':6})
                ax.set_title(title, fontsize=10, fontweight='bold')
                #fig.tight_layout()
                plt.show()
            #First clean the data: Drop rows,columns or impute missing data
            df['outcome'].replace(to_replace=['died','death','Dead','Death','Died','dead'],value='Deceased',inplace=True)
            df['outcome'].replace(to_replace=['recovered','released from quarantine','recovering at home 03.03.2020'],value='Recovered',inplace=True)
            df['outcome'].replace(to_replace=['discharged','Discharged from hospital','discharge'],value='Discharged',inplace=True)
            df['outcome'].replace(to_replace=['Under treatment','treated in an intensive care unit (14.02.2020)','critical condition, intubated as of 14.02.2020','Symptoms only improved with cough. Currently hospitalized for follow-up.','critical condition','severe illness','Critical condition','unstable','severe','Migrated','Migrated_Other'],value='Receiving Treatment',inplace=True)
            df['outcome'].replace(to_replace=['stable','stable condition'],value='Stable',inplace=True)

            #Work only with subset of data where outcome!=Nan
            sub_df = df[df['outcome'].isna()==False]
            sub_df = sub_df[sub_df['outcome']!='https://www.mspbs.gov.py/covid-19.php']
            #Work with subset of data where sex and age is not missing
            smaller_df = sub_df[sub_df['sex'].isna()==False]
            smaller_df = smaller_df[smaller_df['age'].isna()==False]
            smaller_df = smaller_df[smaller_df['date_confirmation'].isna()==False]
            print(smaller_df)
            #Convert dates into dat format
            date_features=['date_onset_symptoms','date_admission_hospital','date_confirmation','travel_history_dates','date_death_or_discharge']
            for feature in date_features:
                smaller_df[feature]=pd.to_datetime(df[feature],format='%d.%m.%Y',errors='coerce')
                smaller_df[feature+'_year']=smaller_df[feature].dt.year 
                smaller_df[feature+'_month']=smaller_df[feature].dt.month
                smaller_df[feature+'_day']=smaller_df[feature].dt.day
                smaller_df[feature+'_dayofweek']=smaller_df[feature].dt.dayofweek
            

            smaller_df['difference_onset_admission']=(smaller_df['date_onset_symptoms'] - smaller_df['date_admission_hospital'])/np.timedelta64(1,'D')
            smaller_df['difference_onset_confirmation']=(smaller_df['date_onset_symptoms'] - smaller_df['date_confirmation'])/np.timedelta64(1,'D')
            smaller_df['difference_travel_onset']=(smaller_df['travel_history_dates'] - smaller_df['date_onset_symptoms'])/np.timedelta64(1,'D')
            smaller_df['difference_onset_deathordischarge']=(smaller_df['date_onset_symptoms'] - smaller_df['date_death_or_discharge'])/np.timedelta64(1,'D')
            smaller_df['difference_confirmation_admission']=(smaller_df['date_confirmation'] - smaller_df['date_admission_hospital'])/np.timedelta64(1,'D')
            smaller_df['difference_travel_admission']=(smaller_df['travel_history_dates'] - smaller_df['date_admission_hospital'])/np.timedelta64(1,'D')
            smaller_df['difference_admission_deathordischarge']=(smaller_df['date_admission_hospital'] - smaller_df['date_death_or_discharge'])/np.timedelta64(1,'D')
            smaller_df['difference_confirmation_deathordischarge']=(smaller_df['date_confirmation'] - smaller_df['date_death_or_discharge'])/np.timedelta64(1,'D')
            smaller_df['difference_travel_confirmation']=(smaller_df['travel_history_dates'] - smaller_df['date_confirmation'])/np.timedelta64(1,'D')#.dt.days

            deceased_binary=[1 if (smaller_df['outcome'][i])=='Deceased' else 0 for i in smaller_df.index]
            smaller_df['deceased_binary']=deceased_binary

            #Sort out the numeric correlations
            numerical_data = list(smaller_df.select_dtypes(include=['int64','float64']).columns)
            numerical_data.remove('latitude')
            numerical_data.remove('longitude')
            corr_df = abs(smaller_df[numerical_data].corr()) 
            df_rank = corr_df.sort_values('deceased_binary',ascending=False)
            print(df_rank)
            df_rank = df_rank[:13]
            print(df_rank)
            for word in df_rank.index:
                for other in df_rank.index:
                    if df_rank[word][other]>0.8 and word!=other and smaller_df[word].isna().sum()<smaller_df[other].isna().sum():#df_rank[word]['deceased_binary']>df_rank[other]['deceased_binary']:
                        print(df_rank[word][other],word,other)
                        print(smaller_df[word].isna().sum(),smaller_df[other].isna().sum())
                        df_rank = df_rank.drop(other)

            print(df_rank)
            for col in smaller_df.columns:
                if ((smaller_df[col].dtype == np.float64 or smaller_df[col].dtype == np.int64) and col not in df_rank.index and col not in ['latitude','longitude']):
                    print(col)
                    smaller_df.drop(col,axis=1,inplace=True)
            print(smaller_df)




            #zero_matrix=[0.0*len(numerical_data) for i in range(len(keywords))]
            #new_df = pd.DataFrame(zero_matrix,columns=numerical_data,index=numerical_data)
            #corr_df = smaller_df.corr()
            #plot_heatmap(corr_df,'Correlation heatmap')

            #differences = ['difference_onset_admission','difference_onset_confirmation','difference_travel_onset','difference_onset_deathordischarge','difference_confirmation_admission',
            #'difference_travel_admission','difference_admission_deathordischarge','difference_confirmation_deathordischarge','difference_travel_confirmation','deceased_binary']
            '''
            differences = []
            for word in list(smaller_df.columns):
                if "difference" in word:
                    differences.append(word)
                
            #word if word in re.compile('differences') for word in list(smaller_df.index)
            #re.findall('differences',list(smaller_df.index))
            differences.extend('deceased_binary')
            diff_corr_df = smaller_df[differences].corr()
            plot_heatmap(diff_corr_df,'Correlation heatmap')

            dates = re.findall('date',list(smaller_df.index))
            dates.extend('deceased_binary')
            dates_corr_df = smaller_df[dates].corr()
            plot_heatmap(corr_df,'Correlation heatmap')
            
            year = re.findall('year',list(smaller_df.index))
            year.extend('deceased_binary')
            dates_corr_df = smaller_df[year].corr()
            plot_heatmap(corr_df,'Correlation heatmap')
            month = re.findall('month',list(smaller_df.index))
            month.extend('deceased_binary')
            dates_corr_df = smaller_df[month].corr()
            plot_heatmap(corr_df,'Correlation heatmap')
            day = re.findall('day',list(smaller_df.index))
            day.extend('deceased_binary')
            dates_corr_df = smaller_df[day].corr()
            plot_heatmap(corr_df,'Correlation heatmap')
'''


    