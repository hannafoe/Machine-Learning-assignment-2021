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

from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier,NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score


#URL for data 
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
            
            print(df.info())
            #print("Age: ",df['age'].unique())

            
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
            
            smaller_df['date_confirmation'+'_year']=smaller_df['date_confirmation'].dt.year 
            smaller_df['date_confirmation'+'_month']=smaller_df['date_confirmation'].dt.month
            smaller_df['date_confirmation'+'_day']=smaller_df['date_confirmation'].dt.day
            smaller_df['date_confirmation'+'_dayofweek']=smaller_df['date_confirmation'].dt.dayofweek
            smaller_df['date_onset_symptoms'+'_year']=smaller_df['date_onset_symptoms'].dt.year 
            smaller_df['date_onset_symptoms'+'_month']=smaller_df['date_onset_symptoms'].dt.month
            smaller_df['date_onset_symptoms'+'_day']=smaller_df['date_onset_symptoms'].dt.day
            smaller_df['date_onset_symptoms'+'_dayofweek']=smaller_df['date_onset_symptoms'].dt.dayofweek
            print(smaller_df.info())

            ##Check which features to drop...
            smaller_df.drop('reported_market_exposure',axis=1,inplace=True)
            smaller_df.drop('sequence_available',axis=1,inplace=True)
            smaller_df.drop('notes_for_discussion',axis=1,inplace=True)
            smaller_df.drop('data_moderator_initials',axis=1,inplace=True)
            smaller_df.drop('ID',axis=1,inplace=True)
            ##Drop all location data except for longitude and latitude
            smaller_df.drop('city',axis=1,inplace=True)
            smaller_df.drop('province',axis=1,inplace=True)
            #smaller_df.drop('country',axis=1,inplace=True)
            smaller_df.drop('geo_resolution',axis=1,inplace=True)
            smaller_df.drop('location',axis=1,inplace=True)
            smaller_df.drop('admin3',axis=1,inplace=True)
            smaller_df.drop('admin2',axis=1,inplace=True)
            smaller_df.drop('admin1',axis=1,inplace=True)
            smaller_df.drop('country_new',axis=1,inplace=True)
            ###
            smaller_df.drop('symptoms',axis=1,inplace=True)
            smaller_df.drop('lives_in_Wuhan',axis=1,inplace=True)
            smaller_df.drop('additional_information',axis=1,inplace=True)
            smaller_df.drop('chronic_disease',axis=1,inplace=True)
            smaller_df.drop('source',axis=1,inplace=True)
            #smaller_df.drop('admin_id',axis=1,inplace=True)
            #Drop those with many nan values
            n = len(smaller_df.index)
            for v in date_features:
                print(v,": ")
                print(smaller_df[v].value_counts())
                print("Nan:",smaller_df[v].isna().sum())
                print("Not nan: ",n-smaller_df[v].isna().sum())
                print()
            #Want to use dates, but there are a lot of missing dates in all the date features except for confirmation date
            #Make different calculations, some with and some without the dates
            #['date_onset_symptoms','date_admission_hospital','date_confirmation','travel_history_dates','date_death_or_discharge']
            
            #smaller_df['difference_onset_admission']=(smaller_df['date_onset_symptoms'] - smaller_df['date_admission_hospital']).dt.days
            #smaller_df['difference_onset_confirmation']=(smaller_df['date_onset_symptoms'] - smaller_df['date_confirmation']).dt.days
            #smaller_df['difference_travel_onset']=(smaller_df['travel_history_dates'] - smaller_df['date_onset_symptoms']).dt.days
            #smaller_df['difference_onset_deathordischarge']=(smaller_df['date_onset_symptoms'] - smaller_df['date_death_or_discharge']).dt.days
            #smaller_df['difference_confirmation_admission']=(smaller_df['date_confirmation'] - smaller_df['date_admission_hospital']).dt.days
            #smaller_df['difference_travel_admission']=(smaller_df['travel_history_dates'] - smaller_df['date_admission_hospital']).dt.days
            #smaller_df['difference_admission_deathordischarge']=(smaller_df['date_admission_hospital'] - smaller_df['date_death_or_discharge']).dt.days
            #smaller_df['difference_confirmation_deathordischarge']=(smaller_df['date_confirmation'] - smaller_df['date_death_or_discharge']).dt.days
            #smaller_df['difference_travel_confirmation']=(smaller_df['travel_history_dates'] - smaller_df['date_confirmation']).dt.days
            
            #cyclical data (dates), change to be able to process it
            dates = ['date_confirmation_month','date_confirmation_day','date_confirmation_dayofweek','date_onset_symptoms_month','date_onset_symptoms_day','date_onset_symptoms_dayofweek'] #except year which is not cyclic
            def encode_cyclic_data(df, feature, max_val): #sin,cos, transformation for cyclic data
                df[feature + '_sin'] = np.sin(2 * np.pi * df[feature]/max_val)
                df[feature + '_cos'] = np.cos(2 * np.pi * df[feature]/max_val)
                return df
            smaller_df = encode_cyclic_data(smaller_df,dates[0],12)
            smaller_df = encode_cyclic_data(smaller_df,dates[1],365)
            smaller_df = encode_cyclic_data(smaller_df,dates[2],7)
            smaller_df = encode_cyclic_data(smaller_df,dates[3],12)
            smaller_df = encode_cyclic_data(smaller_df,dates[4],365)
            smaller_df = encode_cyclic_data(smaller_df,dates[5],7)
            for dat in dates:
                smaller_df.drop(dat,axis=1,inplace=True)

            smaller_df.drop('date_onset_symptoms',axis=1,inplace=True)
            smaller_df.drop('date_admission_hospital',axis=1,inplace=True)
            smaller_df.drop('date_confirmation',axis=1,inplace=True)
            smaller_df.drop('travel_history_dates',axis=1,inplace=True)
            smaller_df.drop('date_death_or_discharge',axis=1,inplace=True)

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
            print("Age: ",smaller_df['age'].unique())

            
            #Create a new column called deceased_binary, target label!!
            deceased_binary=[1 if (smaller_df['outcome'][i])=='Deceased' else 0 for i in smaller_df.index]
            #smaller_df['deceased_binary']=deceased_binary
            #Drop the outcome column, instead we now have the deceased binary
            smaller_df.drop('outcome',axis=1,inplace=True)
            y=pd.Series(deceased_binary)
            
            categorical_data = list(smaller_df.select_dtypes(include=['object','bool']).columns)
            smaller_df[categorical_data] = smaller_df[categorical_data].apply(lambda series: pd.Series(
                LabelEncoder().fit_transform(series[series.notnull()]),
                index=series[series.notnull()].index
            ))
            imp_cat = IterativeImputer(estimator=RandomForestClassifier(), 
                           initial_strategy='most_frequent',
                           max_iter=10, random_state=0)

            smaller_df[categorical_data] = imp_cat.fit_transform(smaller_df[categorical_data])
            ##Split data into test and training set
            X_train, X_test, y_train, y_test = train_test_split(smaller_df, y, test_size=0.2,random_state=0,stratify=y,shuffle=True)
            print(X_train.shape, y_train.shape)
            print(X_test.shape, y_test.shape)

            #Try different imputation techniques
            estimators = [
                BayesianRidge(),
                DecisionTreeRegressor(max_features='sqrt', random_state=0),
                ExtraTreesRegressor(n_estimators=10, random_state=0),
                KNeighborsRegressor(n_neighbors=8)
            ]
            #imputer = IterativeImputer(random_state=0, estimator=BayesianRidge())
            #imputer.fit(X_train)
            #Xtrans = imputer.transform(X_train)
            #print(X_trans)
            #print('Missing: %d' % sum(np.isnan(Xtrans).flatten()))
            
            '''
            imp_num = IterativeImputer(estimator=RandomForestRegressor(),
                                    initial_strategy='mean',
                                    max_iter=10, random_state=0)
            imp_cat = IterativeImputer(estimator=RandomForestClassifier(), 
                                    initial_strategy='most_frequent',
                                    max_iter=10, random_state=0)
            '''


            ##Encode data
            #numerical data
            #numerical_data = X_train[["latitude","longitude","admin_id","difference_onset_admission",
            #"difference_onset_confirmation","difference_travel_onset","difference_onset_deathordischarge",
            #"difference_confirmation_admission","difference_travel_admission","difference_admission_deathordischarge",
            #"difference_confirmation_deathordischarge","difference_travel_confirmation"]]
            numerical_data = list(X_train.select_dtypes(include=['float64','int64']))
            num_pipeline = Pipeline([
                #('imputer', SimpleImputer(strategy = 'most_frequent')),
                ('imputer', IterativeImputer(random_state=0, estimator=KNeighborsRegressor(n_neighbors=10,n_jobs=-1))),
                ('std_scaler', StandardScaler())
            ])

            #categorical data
            categorical_data = list(X_train.select_dtypes(include=['object','bool']).columns)
            
            #trans = [('cat',OneHotEncoder(),categorical_data),('num',num_pipeline,numerical_data)]
            cat_pipeline = Pipeline(steps = [
                ('imputer', SimpleImputer(strategy = 'most_frequent')),
                #('encoder', OneHotEncoder(handle_unknown='ignore')),
                #('encoder', OrdinalEncoder()),
                #('imputer', IterativeImputer(random_state=0, estimator=ExtraTreesRegressor(n_estimators=10, random_state=0))),
                #('imp', SimpleImputer(strategy = 'most_frequent'))
                #('imputer',KNNImputer(n_neighbors=5))
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
                #IterativeImputer(estimator=RandomForestClassifier(), initial_strategy='most_frequent', max_iter=10, random_state=0)
            ])
            full_pipeline = ColumnTransformer([('cat',cat_pipeline,categorical_data),('num',num_pipeline,numerical_data)],remainder='passthrough')
            #Logistic Regression
            log_regression = LogisticRegression(max_iter=500,random_state=0)
            log_pipeline = Pipeline(steps=[('prep',full_pipeline), ('model', log_regression)])
            #print(full_pipeline.fit_transform(X_train))
            log_pipeline.fit(X_train,y_train)
            

            #########################
            #Using prediction pipeline in a grid search
            #param_grid = {
            #    'preprocessor__num__imputer__strategy': ['mean', 'median'],
            #    'classifier__C': [0.1, 1.0, 10, 100],
            #}

            #grid_search = GridSearchCV(clf, param_grid, cv=10)
            #grid_search
            #grid_search.fit(X_train, y_train)

            #print(f"Best params:")
            #print(grid_search.best_params_)
            #print(f"Internal CV score: {grid_search.best_score_:.3f}")
            #cv_results = pd.DataFrame(grid_search.cv_results_)
            #cv_results = cv_results.sort_values("mean_test_score", ascending=False)
            #cv_results[["mean_test_score", "std_test_score",
            #            "param_preprocessor__num__imputer__strategy",
            #            "param_classifier__C"
            #            ]].head(5)
            #print(("best logistic regression from grid search: %.3f"
            #% grid_search.score(X_test, y_test)))
                

            #prediction
            #X_test = full_pipeline.transform(X_test)
            log_prediction_y = log_pipeline.predict(X_test)
            log_score_y = log_pipeline.predict_proba(X_test)
            probs = log_score_y[:,1]
            
            #Use score to get accuracy of model
            log_score=log_pipeline.score(X_test, y_test)
            print("model score: %.5f" % log_score)


            ##
            ##Heat map use one-hot encoding, can use as first step for feature selection!!

            #Compare performance of model on training and test data to see if overfitting
            #Check results of training set with confusion matrix
            log_prediction_y_train = log_pipeline.predict(X_train)
            log_confusion_m_train = pd.crosstab(y_train,log_prediction_y_train,rownames=['Real Values'],colnames=['Predicted Values'],margins=True)
            print(log_confusion_m_train,'\n')
            print("Classification report training set:")
            print(classification_report(y_train, log_prediction_y_train))

            #Check results of testset with confusion matrix
            log_confusion_m = pd.crosstab(y_test,log_prediction_y,rownames=['Real Values'],colnames=['Predicted Values'],margins=True)
            print(log_confusion_m,'\n')
            print("Classification report test set:")
            print(classification_report(y_test, log_prediction_y))
            #roc curve
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
                fig.tight_layout()
                plt.show()
            roc_and_precision_recall(y_test,probs)

            ##rank the probabilities into percentile bins
            
            log_results = pd.DataFrame({'Target':y_test,'class_result':log_prediction_y,'probs':probs})
            log_results['rank']=log_results['probs'].rank(ascending=1).astype(int)
            log_results['rank_pct']=log_results['probs'].rank(ascending=1,pct=True)
            bins = 10
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
                    df_sub = df[(df['rank']>count) & (df['rank']<=bin_maxs[i])]
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

            bin_analysis(log_results,10,True)

            #Feature importance
            feat_imp = permutation_importance(log_pipeline,X_train,y_train,n_repeats=10,random_state=0,n_jobs=-1)
            sorted_feat_imp = feat_imp.importances_mean.argsort()
            f, ax = plt.subplots()
            ax.boxplot(feat_imp.importances[sorted_feat_imp].T,vert=False,labels=X_train.columns[sorted_feat_imp])
            ax.set_title("Permutation Importances of training set")
            f.tight_layout()
            plt.show()

            ##Feature importance analysis
            #df_imp = pd.DataFrame(np.std(X_train, 0), columns=['std'])
            #print(log_regression.coef_)
            #df_imp['coef'] = log_regression.coef_[0]
            #df_imp['coef*std'] = [list(log_regression.coef_[0])[i]*list(np.std(X_train, 0))[i] for i in range(len(df_imp.index))]
            #df_imp['abs(coef*std)'] = np.abs(df_imp['coef*std'])
            #df_imp['log_rank'] = dfimp['abs(coef*std)'].rank(ascending=False)
            #df_rank = df_imp.sort_values('log_rank',ascending=True)
            #print(df_rank)



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

            
            




