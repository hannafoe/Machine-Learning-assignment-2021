# -*- coding: utf-8 -*-


#!pip install eli5

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tarfile
from io import BytesIO
import requests
from requests.exceptions import HTTPError
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest,mutual_info_classif,f_classif
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer,IterativeImputer
from sklearn.metrics import precision_recall_curve,classification_report,roc_curve,mean_squared_error
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from statistics import mean
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.model_selection import GridSearchCV
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV


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
####################################################################################
##PRE-PROCESSING
            #Work only with subset of data where outcome!=Nan
            sub_df = df[df['outcome'].isna()==False]
            sub_df = sub_df[sub_df['outcome']!='https://www.mspbs.gov.py/covid-19.php']
            #Work with subset of data where sex and age is not missing
            smaller_df = sub_df[sub_df['sex'].isna()==False]
            smaller_df = smaller_df[smaller_df['age'].isna()==False]
            smaller_df = smaller_df[smaller_df['date_confirmation'].isna()==False]
            print(smaller_df)
            
            #Clean the data: Drop rows,columns or impute missing data
            smaller_df['outcome'].replace(to_replace=['died','death','Dead','Death','Died','dead'],value='Deceased',inplace=True)
            smaller_df['outcome'].replace(to_replace=['recovered','released from quarantine','recovering at home 03.03.2020'],value='Recovered',inplace=True)
            smaller_df['outcome'].replace(to_replace=['discharged','Discharged from hospital','discharge'],value='Discharged',inplace=True)
            smaller_df['outcome'].replace(to_replace=['Under treatment','treated in an intensive care unit (14.02.2020)','critical condition, intubated as of 14.02.2020','Symptoms only improved with cough. Currently hospitalized for follow-up.','critical condition','severe illness','Critical condition','unstable','severe','Migrated','Migrated_Other'],value='Receiving Treatment',inplace=True)
            smaller_df['outcome'].replace(to_replace=['stable','stable condition'],value='Stable',inplace=True)
            
            ##Do something with different age stuff
            smaller_df['age'].replace(to_replace=['0','1','2','3','4','5','6','7','8','9','0-10','0-6','1.75','0.6666666667','0.5','0.25','0.58333','0.08333','6 weeks','0.75',0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,'1.5','0.4','0.3','2.5','0.2','0.7','0.1','3.5','0.9','0.6','0-1','18 months','18 month','7 months','4 months','13 month','5 months','8 month','6 months','9 month','5 month','11 month','5-14','0-4','00-04','05-14','5-9'],value='0-9',inplace=True)
            smaller_df['age'].replace(to_replace=['10','11','12','13','14','15','16','17','18','19','13-19','0-18','16-17','12-19','0-19','18-20','0-20','14-18',11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,'10-14','15-19','11-12',10.0],value='10-19',inplace=True)
            smaller_df['age'].replace(to_replace=['20','21','22','23','24','25','26','27','28','29','20-30','20-39','21-39','1-42','23-24','26-27',21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,'29.6','15-34','20-24','25-29','27-29','22-23',20.0],value='20-29',inplace=True)
            smaller_df['age'].replace(to_replace=['30','31','32','33','34','35','36','37','38','39','5-59','18-50','18-49','30-40','0-60','30-35','23-72','36-45','14-60','13-65','4-64','20-57','34-44''5-56','28-35',31.0,32.0,33.0,34.0,35.0,36.0,37.0,38.0,39.0,'30-34','35-39','27-40','37-38',30.0],value='30-39',inplace=True)
            smaller_df['age'].replace(to_replace=['40','41','42','43','44','45','46','47','48','49','40-49','19-65','40-50','40-45','17-66','20-69','19-77','13-69','21-72','30-60','8-68','17-65','19-75','21-61','22-60','2-87','23-71','27-58','25-59','48-49','40-41',41.0,42.0,43.0,44.0,45.0,46.0,47.0,48.0,49.0,40.0,'35-59','45-49','40-44','18-65','35-54'],value='40-49',inplace=True)
            smaller_df['age'].replace(to_replace=['50','51','52','53','54','55','56','57','58','59','50-69','38-68','41-60','18-60','40-69','30-69','54-56','18-99','34-66','22-80','18 - 100','33-78','16-80','11-80','30-61','22-66','39-77',51.0,52.0,53.0,54.0,55.0,56.0,57.0,58.0,59.0,50.0,'54.9','50-54','55-59','30-70','20-70','50-60'],value='50-59',inplace=True)
            smaller_df['age'].replace(to_replace=['60','61','62','63','64','65','66','67','68','69','40-89','60-70','55-74','25-89','15-88',61.0,62.0,63.0,64.0,65.0,66.0,67.0,68.0,69.0,'60-64','65-69', '18-','23-84',60.0],value='60-69',inplace=True)
            smaller_df['age'].replace(to_replace=['70','71','72','73','74','75','76','77','78','79','70-70','60-','61-80','50-100','50-','70-82',71.0,72.0,73.0,74.0,75.0,76.0,77.0,78.0,79.0,'60-79','75-79','70-74','55-','74-76',70.0],value='70-79',inplace=True)
            smaller_df['age'].replace(to_replace=['80','81','82','83','84','85','86','87','88','89','80-80','65-99','60-99','50-99','65-','75-','70-100','60-100','87-88',81.0,82.0,83.0,84.0,85.0,86.0,87.0,88.0,89.0,'80-84',80.0],value='80-89',inplace=True)
            smaller_df['age'].replace(to_replace=['90','91','92','93','94','95','96','97','98','99','100','101','102','103','104','107','121','90-99',91.0,92.0,93.0,94.0,95.0,96.0,97.0,98.0,99.0,100.0,101.0,102.0,103.0,104.0,105.0,106.0,107.0,108.0,109.0,'80-','80+','85+','105',90.0],value='90+',inplace=True)
            #print("Age: ",smaller_df['age'].unique())
            
            #Convert dates into dat format
            date_features=['date_onset_symptoms','date_admission_hospital','date_confirmation','travel_history_dates','date_death_or_discharge']
            for feature in date_features:
                smaller_df[feature]=pd.to_datetime(df[feature],format='%d.%m.%Y',errors='coerce')
                smaller_df[feature+'_year']=smaller_df[feature].dt.year 
                smaller_df[feature+'_month']=smaller_df[feature].dt.month
                smaller_df[feature+'_day']=smaller_df[feature].dt.day
                smaller_df[feature+'_dayofweek']=smaller_df[feature].dt.dayofweek
            
            smaller_df['difference_admission_onset']=(smaller_df['date_admission_hospital'] - smaller_df['date_onset_symptoms']).dt.days
            smaller_df['difference_confirmation_onset']=(smaller_df['date_confirmation'] - smaller_df['date_onset_symptoms']).dt.days
            smaller_df['difference_onset_travel']=(smaller_df['date_onset_symptoms'] - smaller_df['travel_history_dates']).dt.days
            smaller_df['difference_deathordischarge_onset']=(smaller_df['date_death_or_discharge'] - smaller_df['date_onset_symptoms']).dt.days
            smaller_df['difference_admission_confirmation']=(smaller_df['date_admission_hospital'] - smaller_df['date_confirmation']).dt.days
            smaller_df['difference_admission_travel']=(smaller_df['date_admission_hospital'] - smaller_df['travel_history_dates']).dt.days
            smaller_df['difference_deathordischarge_admission']=(smaller_df['date_death_or_discharge'] - smaller_df['date_admission_hospital']).dt.days
            smaller_df['difference_deathordischarge_confirmation']=(smaller_df['date_death_or_discharge'] - smaller_df['date_confirmation']).dt.days
            smaller_df['difference_confirmation_travel']=(smaller_df['date_confirmation'] - smaller_df['travel_history_dates']).dt.days
            
            #Create target column: Deceased
            deceased_binary=[1 if (smaller_df['outcome'][i])=='Deceased' else 0 for i in smaller_df.index]
            smaller_df['deceased_binary']=deceased_binary
            smaller_df.drop('outcome',axis=1,inplace=True)
            #Drop all features that are not relevant
            smaller_df.drop('source',axis=1,inplace=True)
            smaller_df.drop('ID',axis=1,inplace=True)
            smaller_df.drop('data_moderator_initials',axis=1,inplace=True)
            #Drop additional information, since it is hard to interpret in a large basis and hard to impute
            smaller_df.drop('additional_information',axis=1,inplace=True)
            #Drop country, province and such id
            smaller_df.drop('city',axis=1,inplace=True)
            smaller_df.drop('province',axis=1,inplace=True)
            smaller_df.drop('country',axis=1,inplace=True)
            smaller_df.drop('geo_resolution',axis=1,inplace=True)
            smaller_df.drop('country_new',axis=1,inplace=True)
            smaller_df.drop('admin_id',axis=1,inplace=True)
            #Create a new column which combines chronic disease+symptom (onset_symptom or symptoms)
            smaller_df['symptoms_binary'] = [1 if (not (pd.isna(smaller_df['date_onset_symptoms'][i])) or not (pd.isna(smaller_df['symptoms'][i]))) else 0 for i in smaller_df.index]
            smaller_df['chronic_disease_binary'] = [1 if (smaller_df['chronic_disease_binary'][i]==1 or not (pd.isna(smaller_df['chronic_disease'][i]))) else 0 for i in smaller_df.index]
            smaller_df['symptom_disease_binary'] = [1 if (smaller_df['chronic_disease_binary'][i]==1 or smaller_df['symptoms_binary'][i]==1) else 0 for i in smaller_df.index]
            smaller_df['travel_history_binary'] = [1 if (not (pd.isna(smaller_df['travel_history_dates'][i])) or smaller_df['travel_history_binary'][i]==1 or not (pd.isna(smaller_df['travel_history_location'][i]))) else 0 for i in smaller_df.index]
            #smaller_df['country'] = [smaller_df['country'][i] if not (pd.isna(smaller_df['country'][i])) else smaller_df['country_new'][i] for i in smaller_df.index]
            #smaller_df.drop('country_new',axis=1,inplace=True)
            #Drop all columns with too many nan values
            n = len(smaller_df.index)
            for v in smaller_df.columns:
                if n-smaller_df[v].isna().sum()<200: #not nan values<200
                    smaller_df.drop(v,axis=1,inplace=True)
                    print(v)#,n-smaller_df[v].isna().sum())
                elif (smaller_df[v].dtype!=np.float64 and smaller_df[v].dtype!=np.int64):
                    if n-smaller_df[v].isna().sum()<5000:
                        smaller_df.drop(v,axis=1,inplace=True)
                        print(v)
            print(smaller_df.columns)
            
            #Sort out the numeric correlations
            def plot_heatmap(df,title):
                plt.subplots(figsize=(10,10))
                sns.heatmap(df)
                plt.show()
            numerical_data = list(smaller_df.select_dtypes(include=['int64','float64']).columns)
            numerical_data.remove('latitude')
            numerical_data.remove('longitude')
            #Create pearson correlation dataframe
            corr_df = abs(smaller_df[numerical_data].corr()) 
            plot_heatmap(corr_df,'Pearson correlation heatmap of all numeric data')
            #Sort features according absolute pearson correlation
            df_rank = corr_df.sort_values('deceased_binary',ascending=False)
            #Take the top 13 features best correlated with deceased binary
            df_rank = df_rank[:13]
            #Remove one of two features if they are correlated more than 0.8 to each other
            #Remove the feature with more nan values
            for word in df_rank.index:
                for other in df_rank.index:
                    if df_rank[word][other]>0.8 and word!=other and smaller_df[word].isna().sum()<smaller_df[other].isna().sum():
                        df_rank = df_rank.drop(other)
            print(df_rank)
            best_num_features = list(df_rank.index)
            best_num_features.extend(['latitude','longitude'])
            #Drop all features not in best_num_features
            for col in smaller_df.columns:
                if ((smaller_df[col].dtype == np.float64 or smaller_df[col].dtype == np.int64) and col not in best_num_features):
                    smaller_df.drop(col,axis=1,inplace=True)
            print(smaller_df)
            #Now we can drop all dat features
            dat_features = list(smaller_df.select_dtypes(include=['datetime64[ns]']).columns)
            for v in dat_features:
                smaller_df.drop(v,axis=1,inplace=True)
            print(best_num_features)
            corr_df = abs(smaller_df[best_num_features].corr()) 
            plot_heatmap(corr_df,'Pearson correlation heatmap of best numeric data')
            
            #Now deal with categorical data
            #sort out the categorical correlations
            categorical_data = list(smaller_df.select_dtypes(include=['object','bool']).columns)
            print(categorical_data)
            #Label Encode all data
            #mappings_dict will be later used to convert the label encoded features 
            #back to its original values, to understand feature importance later
            mappings_dict = {}
            label_encoder = LabelEncoder()
            for col in categorical_data:
                smaller_df[col] = pd.Series(label_encoder.fit_transform(smaller_df[col][smaller_df[col].notna()]),index=smaller_df[col][smaller_df[col].notna()].index)
                mappings_dict[col]=dict(zip(label_encoder.transform(label_encoder.classes_),label_encoder.classes_))
                
            #Impute all categorical data
            imp_cat = IterativeImputer(estimator=RandomForestClassifier(max_depth=5,n_jobs=-1), 
                            initial_strategy='most_frequent',
                            max_iter=10, random_state=0)
            smaller_df[categorical_data] = imp_cat.fit_transform(smaller_df[categorical_data])
            print(smaller_df[categorical_data])
            #for col in categorical_data:
            #    print('Missing: %d' % sum(smaller_df[col].isna()))
            y = smaller_df['deceased_binary']
            y = y.astype(int)
            X = pd.DataFrame(smaller_df[categorical_data])
            #Select the best data
            #Since there aren't many categorical features, just see feature importance
            #First technique: mutual_info_classif
            cat_selection = SelectKBest(score_func=mutual_info_classif, k='all')
            cat_selection.fit_transform(X,y)
            for feature in range(len(cat_selection.scores_)):
                print('Feature %s: %f' % (categorical_data[feature], cat_selection.scores_[feature]))
            # plot the scores
            fig, ax = plt.subplots(figsize=(10,10))
            ax = sns.barplot(x=[categorical_data[i] for i in range(len(cat_selection.scores_))], y=cat_selection.scores_)
            ax.set_title("Categorical feature selection: mutual_info_classif", fontsize=10, fontweight='bold')
            plt.xlabel("features")
            plt.ylabel("scores")
            plt.show()
            cat_df = pd.DataFrame({
                'features':categorical_data,
                'mutual_info_classif':cat_selection.scores_
                })
            cat_df = cat_df.sort_values('mutual_info_classif',ascending=False)
            cat_df['rank_mutual_info_classif']=[i for i in range(len(cat_df))]
            #Second technique: f_classif
            cat_selection = SelectKBest(score_func=f_classif, k='all')
            cat_selection.fit_transform(X,y)
            for feature in range(len(cat_selection.scores_)):
                print('Feature %s: %f' % (categorical_data[feature], cat_selection.scores_[feature]))
            fig, ax = plt.subplots()
            ax = sns.barplot(x=[categorical_data[i] for i in range(len(cat_selection.scores_))], y=cat_selection.scores_)
            ax.set_title("Categorical feature selection: f_classif", fontsize=10, fontweight='bold')
            plt.xlabel("features")
            plt.ylabel("scores")
            fig.tight_layout()
            plt.show()
            cat_df['f_classif'] = cat_selection.scores_
            cat_df = cat_df.sort_values('f_classif',ascending=False)
            cat_df['rank_f_classif']=[i for i in range(len(cat_df))]
            #Not really needed: code still from when more categorical features were used to test
            #Those that are ranked high in both, add rankings up and divide by two
            cat_df['rank']=cat_df['rank_mutual_info_classif']+cat_df['rank_f_classif']
            cat_df = cat_df.sort_values('rank',ascending=True)
            print(cat_df)
            best_cat_features = list(cat_df['features'].iloc[:10])
            if 'age' not in best_cat_features:
                best_cat_features.append('age')
            if 'sex' not in best_cat_features:
                best_cat_features.append('sex')
            print(best_cat_features)
            for col in categorical_data:
                if col not in best_cat_features:
                    smaller_df.drop(col,axis=1,inplace=True)
            #Order the columns such that first all categorical features, then all numerical features
            smaller_df = pd.concat([smaller_df[best_cat_features], smaller_df[best_num_features]], axis=1, join="inner")
            
#######################################################################################################
            ##Split data into test and training set
            #Drop the deceased_binary column
            smaller_df.drop('deceased_binary',axis=1,inplace=True)
            best_num_features.remove('deceased_binary')
            
            X_train, X_test, y_train, y_test = train_test_split(smaller_df, y, test_size=0.2,random_state=0,stratify=y,shuffle=True)
            print(X_train.shape, y_train.shape)
            print(X_test.shape, y_test.shape)
            
            numerical_data = best_num_features
            num_pipeline = Pipeline([
                ('imputer', IterativeImputer(random_state=0, estimator=KNeighborsRegressor(n_neighbors=5,n_jobs=-1),max_iter=100)),
                ('std_scaler', StandardScaler())
            ])
            
            #categorical data
            categorical_data = best_cat_features
            one = OneHotEncoder(handle_unknown='ignore')
            cat_pipeline = Pipeline(steps = [
                ('imputer', SimpleImputer(strategy = 'most_frequent')),#not really necessary, since data was already imputed, but keep it there anyway
                ('encoder', one)
            ])
            full_pipeline = ColumnTransformer([('cat',cat_pipeline,categorical_data),('num',num_pipeline,numerical_data)],remainder='passthrough')
            X_train = full_pipeline.fit_transform(X_train)
            X_test = full_pipeline.transform(X_test)
####################################################################################
##TRAINING
            #Logistic Regression
            
            log_grid = {'C': [2,3,4,5,6,7], 'solver':['lbfgs','liblinear']}
            log_regression = LogisticRegression(max_iter=500,random_state=0,n_jobs=-1,verbose=2)
            
            # Cross validated grid search
            log_grid_search = GridSearchCV(log_regression, log_grid, return_train_score=True,cv=7)
            
            # Fit the model
            log_grid_search.fit(X_train, y_train)
            print("log regression best score:", log_grid_search.best_score_)
            print("log regression best estimator:",log_grid_search.best_estimator_)
            
            log_regression = log_grid_search.best_estimator_
            log_regression.fit(X_train,y_train)
            
            #prediction
            log_prediction_y_test = log_regression.predict(X_test)
            log_score_y = log_regression.predict_proba(X_test)
            log_probs = log_score_y[:,1]
            log_prediction_y_train = log_regression.predict(X_train)
            
            #Use score to get accuracy of model
            log_score=log_regression.score(X_test, y_test)
            print("model score: %.5f" % log_score)
            
            ############################################################################
            
            #This version can be used while commenting out the upper part, if there is not enough time or memory
            #to do the grid search
            #log_regression = LogisticRegression(C=4, max_iter=500, random_state=0,verbose=2,n_jobs=-1)
            #log_regression.fit(X_train,y_train)
            #prediction
            #log_prediction_y_test = log_regression.predict(X_test)
            #log_score_y = log_regression.predict_proba(X_test)
            #log_probs = log_score_y[:,1]
            #log_prediction_y_train = log_regression.predict(X_train)
            
            #Use score to get accuracy of model
            #log_score=log_regression.score(X_test, y_test)
            #print("model score: %.5f" % log_score)
            ############################################################################
            #KNN Classification
                        
            knn = KNeighborsClassifier()
            knn_grid = {'n_neighbors': [4,5,6]}
            # Cross validated grid search to find best n_neighbors
            knn_grid_search = GridSearchCV(knn,knn_grid,return_train_score=True,cv=5)
            # Fit the model
            knn_grid_search.fit(X_train, y_train)
            print("knn best score:", knn_grid_search.best_score_)
            print("knn best estimator:",knn_grid_search.best_estimator_)
            knn = knn_grid_search.best_estimator_
            ###########################################################################
            
            #This version can be used while commenting out the upper part, if there is not enough time or memory
            #to do the grid search
            #knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
            #                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
            #                     weights='uniform')
            #knn.fit(X_train,y_train)
            ############################################################################
            #prediction
            knn_prediction_y_test = knn.predict(X_test)
            knn_score_y = knn.predict_proba(X_test)
            knn_probs = knn_score_y[:,1]
            knn_prediction_y_train = knn.predict(X_train)
            #Use score to get accuracy of model
            knn_score=knn.score(X_test,y_test)
            print("model score: %.5f" % knn_score)
            #############################################################################
            #Random Forest Classification
            rf = RandomForestClassifier(random_state=0,n_jobs=-1)
            #Features that I want to select:
            n_estimators = [x*10 for x in range(20,200)] #number of trees
            bootstrap = [True,False] #sample selection method
            min_samples_split = [2,4,6] #number of samples to split node
            max_depth = [5,10,20,30,40]
            #Grid for random search
            rf_random_grid = {'n_estimators':n_estimators,'bootstrap':bootstrap,'min_samples_split':min_samples_split,'max_depth':max_depth}
            rf_random_grid_search = RandomizedSearchCV(rf,rf_random_grid,cv=3,random_state=0,n_jobs=-1,n_iter=10,verbose=2)
            rf_random_grid_search.fit(X_train,y_train)
            print("rf random best score:", rf_random_grid_search.best_score_)
            print("rf random best estimator:",rf_random_grid_search.best_estimator_)
            
            rf = RandomForestClassifier(random_state=0,n_jobs=-1,min_samples_split=2,max_features='auto')
            #rf_grid = {'n_estimators':[100,500,1000],'bootstrap':[True,False],'max_depth':[40,60]}
            rf_grid = {'n_estimators':[90,100,110],'max_depth':[40,50,60]}
            # Cross validated grid search to find better parameters
            rf_grid_search = GridSearchCV(rf,rf_grid,return_train_score=True,cv=3,n_jobs=-1,verbose=2)
            # Fit the model
            rf_grid_search.fit(X_train, y_train)
            print("rf best score:", rf_grid_search.best_score_)
            print("rf best estimator:",rf_grid_search.best_estimator_)
            rf = rf_grid_search.best_estimator_
            ##########################################################################
            #This version can be used while commenting out the upper part, if there is not enough time or memory
            #to do the grid search
            #rf = RandomForestClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
            #                       criterion='gini', max_depth=50, max_features='auto',
            #                       max_leaf_nodes=None, max_samples=None,
            #                       min_impurity_decrease=0.0, min_impurity_split=None,
            #                       min_samples_leaf=1, min_samples_split=2,
            #                       min_weight_fraction_leaf=0.0, n_estimators=100,
            #                       n_jobs=-1, oob_score=False, random_state=0, verbose=0,
            #                       warm_start=False)
            #rf.fit(X_train,y_train)
            ##########################################################################
            #prediction
            rf_prediction_y_test = rf.predict(X_test)
            rf_score_y = rf.predict_proba(X_test)
            rf_probs = rf_score_y[:,1]
            rf_prediction_y_train = rf.predict(X_train)
            
            #Use score to get accuracy of model
            rf_score=rf.score(X_test,y_test)
            print("model score: %.5f" % rf_score)
#########################################################################################
##PERFORMANCE ANALYSIS            
            #Feature importance analysis
            def feature_importance_analysis(models,X_test,y_test,perm):
                one_hot_columns = list(full_pipeline.named_transformers_['cat'].named_steps['encoder'].get_feature_names(input_features=categorical_data))
                final_columns=[]
                for col in one_hot_columns:
                    for feature in categorical_data:
                        if feature in col:
                            if feature=="country" and "country_new" in col:
                                continue
                            ending = int(float(col[(len(feature)+1):]))
                            if mappings_dict[feature][ending]=='nan':
                                print(ending,col[:(len(feature))])
                                print(mappings_dict[feature][ending])
                            final_columns.append(str(col[:(len(feature))])+"_"+str(mappings_dict[feature][ending]))
                final_columns.extend(numerical_data)
                for model in models:
                    print(model)
                    print(eli5.explain_weights_df(model,feature_names=final_columns,top=40))
                    if perm==True:
                        #perm = PermutationImportance(model, random_state=1,cv="prefit",scoring="balanced_accuracy").fit(X_test.toarray(), y_test)
                        perm = PermutationImportance(model, random_state=1,cv="prefit",scoring="balanced_accuracy").fit(X_test, y_test)
                        print(eli5.explain_weights_df(perm,feature_names=final_columns,top=40))
                        print(perm.feature_importances_)
            feature_importance_analysis([log_regression,knn,rf],X_test,y_test,True)
            
            #Compare performance of model on training and test data to see if overfitting
            #Check results of training set with confusion matrix
            
            def performance_report(models,y_train,y_test,y_preds_train,y_preds_test):
                for i in range(len(models)):
                    confusion_mat_train = pd.crosstab(y_train,y_preds_train[i],rownames=['Real Values'],colnames=['Predicted Values'],margins=True)
                    print(confusion_mat_train,'\n')
                    print("Classification report training set ",models[i],":")
                    print(classification_report(y_train, y_preds_train[i]))
            
                    #Check results of testset with confusion matrix
                    confusion_mat_test = pd.crosstab(y_test,y_preds_test[i],rownames=['Real Values'],colnames=['Predicted Values'],margins=True)
                    print(confusion_mat_test,'\n')
                    print("Classification report test set ",models[i],":")
                    print(classification_report(y_test, y_preds_test[i]))
                    #Mean squared error
                    mse = mean_squared_error(y_test, y_preds_test[i])
                    print("Mean squared error ",models[i],":",mse)
                    #Confidence interval
                    #Method for confidence interval taken from
                    #https://colab.research.google.com/github/ageron/handson-ml2/blob/master/02_end_to_end_machine_learning_project.ipynb#scrollTo=Bziq7SvDmuqf
                    sqe_list = (y_preds_test[i] - y_test)**2 #squared error list
                    mean_sqe = mean(sqe_list)
                    z_score = stats.norm.ppf(1.95 / 2)
                    z_margin = z_score * sqe_list.std(ddof=1)/np.sqrt(len(y_test))
                    print("Confidence Interval: ",np.sqrt(mean_sqe - z_margin), np.sqrt(mean_sqe + z_margin))
            performance_report([log_regression,knn,rf],y_train,y_test,[log_prediction_y_train,knn_prediction_y_train,rf_prediction_y_train],[log_prediction_y_test,knn_prediction_y_test,rf_prediction_y_test])
            
            #roc curve
            #idea taken from
            #https://colab.research.google.com/drive/1isPQd9LJ8Xc7A9u-OrxoU3f4wo6vkqlo?usp=sharing
            def roc_and_precision_recall(y,score_y):
                fpr, tpr, thresh = roc_curve(y, score_y)
                precision, recall, thresholds = precision_recall_curve(y, score_y)
                average_precision = average_precision_score(y,score_y)
                fig, axs = plt.subplots(1,2)
                axs[0].plot(fpr,tpr,color='orange',label='ROC curve')
                axs[0].plot([0,1],[0,1],color='navy',linestyle='--')
                axs[0].set_title('ROC Curve')
                axs[0].set(xlabel='FPR',ylabel='TPR')
                axs[0].legend(loc="lower right")
                axs[1].plot(recall,precision,color='navy',label='Precision recall curve')
                axs[1].set_title('Precision-Recall curve')
                axs[1].set(xlabel='Recall',ylabel='Precision')
                axs[1].legend(loc="lower left")
                fig.suptitle('Average precision-recall score: {0:0.2f}'.format(average_precision))
                #fig.tight_layout()
                plt.show()
            roc_and_precision_recall(y_test,log_probs)
            roc_and_precision_recall(y_test,knn_probs)
            roc_and_precision_recall(y_test,rf_probs)
            
            ##rank the probabilities into percentile bins
            
            log_results = pd.DataFrame({'Target':y_test,'class_result':log_prediction_y_test,'probs':log_probs})
            log_results['rank']=log_results['probs'].rank(ascending=1).astype(int)
            log_results['rank_pct']=log_results['probs'].rank(ascending=1,pct=True)
            
            bins = 5
            #idea taken from
            #https://colab.research.google.com/drive/1isPQd9LJ8Xc7A9u-OrxoU3f4wo6vkqlo?usp=sharing
            def bin_analysis(df,bin_num,plot):
                cols = ['min_rank','max_rank','min_prob','max_prob','num_pos_instances','bin_size','pos_rate']
                roc_cols = ['min_prob','max_prob','Sensitivity (TPR)','FNR','FPR','Specificity (TNR)']
                num_ins = len(df.index)
                bin_df = pd.DataFrame(columns=cols)
                roc_df = pd.DataFrame(columns=roc_cols)
                count = 0
                bin_size = int(num_ins/bin_num)
                c=0
                bin_maxs=[]
                for i in range(bin_num):
                    if (i%10)<int((num_ins/bin_num-bin_size)*10):
                        c+=bin_size+1
                        bin_maxs.append(c)
                    else:
                        c+=bin_size
                        bin_maxs.append(c)
                bin_maxs[bin_num-1]=num_ins
                for i in range(bin_num):
                    df_sub = df[(df['rank']>=count) & (df['rank']<bin_maxs[i])]
                    if len(df_sub)==0:
                      extra_count=bin_maxs[i]+1
                      while (len(df_sub)==0 and extra_count<=num_ins):
                        df_sub = df[(df['rank']>=count) & (df['rank']<extra_count)]
                        extra_count+=1
                      if i==0:
                        l = bin_maxs[i]
                      else:
                        l = bin_maxs[i]-bin_maxs[i-1]
                      df_sub = df_sub[:l]
                    count+=df_sub.shape[0]
                    num_pos_instances=len(df_sub[df_sub['Target']==1])
                    bin_df.loc[i]=([count-df_sub.shape[0],count,min(df_sub['probs']),max(df_sub['probs']),num_pos_instances,df_sub.shape[0],num_pos_instances/df_sub.shape[0]])
                    sub_df_sm = df[df['rank']<bin_maxs[i]]
                    sub_df_l = df[df['rank']>=bin_maxs[i]]
                    TP = len(sub_df_l[sub_df_l['Target']==1])
                    FN = len(sub_df_sm[sub_df_sm['Target']==1])
                    TPR = TP/(TP+FN)
                    FP = len(sub_df_l.index)-TP
                    TN = len(sub_df_sm.index)-FN
                    FPR = FP/(FP+TN)
                    roc_df.loc[i]=[min(df_sub['probs']),max(df_sub['probs']),TPR,1-TPR,FPR,1-FPR]
                print("Positive counts per bin: ")
                print(bin_df)
                pos =len(df[df['Target']==1])
                print("Overall: ")
                print("Number of positive instances: ", pos)
                print("Number of instances: ",len(df.index))
                print("Positive rate: ",pos/len(df.index))
                if plot==True:
                    fig, ax = plt.subplots()
                    ax = sns.barplot(x=bin_df.index,y='pos_rate',data=bin_df)
                    ax.set_title("Positive rate per bin", fontsize=10, fontweight='bold')
                    plt.xlabel("Bins")
                    plt.ylabel("Positive rate")
                    fig.tight_layout()
                    plt.show()
                print("ROC stats: ")
                print(roc_df)
            bin_analysis(log_results,bins,True)
