# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 18:33:59 2021

@author: ich
"""

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
from sklearn.feature_selection import SelectKBest,chi2,mutual_info_classif,f_classif
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer,IterativeImputer
from sklearn.metrics import precision_recall_curve,classification_report,roc_curve, auc,mean_squared_error
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
from sklearn.model_selection import GridSearchCV
from scipy import stats
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


file = './data.xlsx'
smaller_df = pd.read_excel(file)
###########################################################################################
##Split data into test and training set
#Drop the deceased_binary column
best_cat_features=['source', 'age', 'country_new', 'country', 'city', 'data_moderator_initials', 'geo_resolution', 'province', 'chronic_disease_binary', 'sex']
best_num_features=['deceased_binary', 'difference_deathordischarge_confirmation', 'difference_deathordischarge_onset', 'admin_id', 'date_confirmation_month', 'date_admission_hospital_month', 'travel_history_dates_day', 'difference_admission_confirmation', 'travel_history_dates_dayofweek', 'date_admission_hospital_day', 'date_death_or_discharge_dayofweek', 'latitude', 'longitude']
y = smaller_df['deceased_binary']
smaller_df.drop('deceased_binary',axis=1,inplace=True)
best_num_features.remove('deceased_binary')

X_train, X_test, y_train, y_test = train_test_split(smaller_df, y, test_size=0.2,random_state=0,stratify=y,shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.02, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(X_val,y_val)

numerical_data = best_num_features
num_pipeline = Pipeline([
    #('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('imputer', IterativeImputer(random_state=0, estimator=KNeighborsRegressor(n_neighbors=10,n_jobs=-1))),
    ('std_scaler', StandardScaler())
])

#categorical data
categorical_data = best_cat_features#list(X_train.select_dtypes(include=['object','bool']).columns)
one = OneHotEncoder(handle_unknown='ignore')
cat_pipeline = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('encoder', one)
])
full_pipeline = ColumnTransformer([('cat',cat_pipeline,categorical_data),('num',num_pipeline,numerical_data)],remainder='passthrough')
X_train = full_pipeline.fit_transform(X_train)
X_test = full_pipeline.transform(X_test)
#######################################################################
#Logistic Regression
'''
log_grid = {'C': [4,5,6,7,8,10], 'solver':['lbfgs','liblinear']}#'C': [4,5,6,7,8,9,10]
log_regression = LogisticRegression(max_iter=500,random_state=0)

#clf_lr = LogisticRegression(class_weight='balanced', dual=False, 
#          fit_intercept=True, intercept_scaling=1, max_iter=200,
#          n_jobs=1, random_state=0, tol=0.0001, verbose=0, warm_start=False)

# Cross validated grid search
log_grid_search = GridSearchCV(log_regression, log_grid, return_train_score=True,cv=10)

# Fit the model
log_grid_search.fit(X_train, y_train)
print("log regression best score:", log_grid_search.best_score_)
print("log regression best estimator:",log_grid_search.best_estimator_)

log_regression = log_grid_search.best_estimator_
log_regression = LogisticRegression(C=4, max_iter=500, random_state=0)
log_regression.fit(X_train,y_train)

#log_pipeline = Pipeline(steps=[('prep',full_pipeline), ('model', log_regression)])
####################################################################################
#prediction

log_prediction_y_test = log_regression.predict(X_test)
log_score_y = log_regression.predict_proba(X_test)
log_probs = log_score_y[:,1]
log_prediction_y_train = log_regression.predict(X_train)

#Use score to get accuracy of model
log_score=log_regression.score(X_test, y_test)
print("model score: %.5f" % log_score)
'''
############################################################################
#KNN Classification

knn = KNeighborsClassifier()
knn_grid = {'n_neighbors': [4,5]}
# Cross validated grid search to find best n_neighbors
knn_grid_search = GridSearchCV(knn,knn_grid,return_train_score=True,cv=3,n_jobs=-1)
# Fit the model
knn_grid_search.fit(X_train, y_train)
print("knn best score:", knn_grid_search.best_score_)
print("knn best estimator:",knn_grid_search.best_estimator_)
knn = knn_grid_search.best_estimator_
#knn = KNeighborsClassifier(n_neighbors=6)
#knn.fit(X_train,y_train)

knn_prediction_y_test = knn.predict(X_test)
knn_score_y = knn.predict_proba(X_test)
knn_probs = knn_score_y[:,1]
knn_prediction_y_train = knn.predict(X_train)

#Use score to get accuracy of model
knn_score=knn.score(X_test,y_test)
print("model score: %.5f" % knn_score)
knn_acc = accuracy_score(y_true=y_test, y_pred=knn_prediction_y_test)
print("accuracy score: %.5f" % knn_acc)

############################################################################
#Feature importance analysis
def feature_importance_analysis(models,X_val,y_val,perm):
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
    X_val = full_pipeline.transform(X_val)
    for model in models:
        print(eli5.explain_weights_df(model,feature_names=final_columns,top=30))
        if perm==True:
            perm = PermutationImportance(model, random_state=1,cv="prefit",scoring="balanced_accuracy").fit(X_val.toarray(), y_val)
            print(eli5.explain_weights_df(perm,feature_names=final_columns))
            print(perm.feature_importances_)
#feature_importance_analysis([knn],X_val,y_val,False)
#feature_importance_analysis([log_regression,knn],X_val,y_val,False)
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
        #https://colab.research.google.com/github/ageron/handson-ml2/blob/master/02_end_to_end_machine_learning_project.ipynb#scrollTo=Bziq7SvDmuqf
        sqe_list = (y_preds_test[i] - y_test)**2 #squared error list
        mean_sqe = mean(sqe_list)
        z_score = stats.norm.ppf(1.95 / 2)
        z_margin = z_score * sqe_list.std(ddof=1)/np.sqrt(len(y_test))
        print("Confidence Interval: ",np.sqrt(mean_sqe - z_margin), np.sqrt(mean_sqe + z_margin))
performance_report([knn],y_train,y_test,[knn_prediction_y_train],[knn_prediction_y_test])
#performance_report([log_regression,knn],y_train,y_test,[log_prediction_y_train,knn_prediction_y_train],[log_prediction_y_test,knn_prediction_y_test])
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
#roc_and_precision_recall(y_test,log_probs)
roc_and_precision_recall(y_test,knn_probs)

##rank the probabilities into percentile bins
'''
log_results = pd.DataFrame({'Target':y_test,'class_result':log_prediction_y_test,'probs':log_probs})
log_results['rank']=log_results['probs'].rank(ascending=1).astype(int)
log_results['rank_pct']=log_results['probs'].rank(ascending=1,pct=True)'''
knn_results = pd.DataFrame({'Target':y_test,'class_result':knn_prediction_y_test,'probs':knn_probs})
knn_results['rank']=knn_results['probs'].rank(ascending=1).astype(int)
knn_results['rank_pct']=knn_results['probs'].rank(ascending=1,pct=True)
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

#bin_analysis(log_results,bins,True)
bin_analysis(knn_results,bins,True)