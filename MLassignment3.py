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


file = './data.xlsx'
smaller_df = pd.read_excel(file)
smaller_df.drop('Unnamed: 0',axis=1,inplace=True)
smaller_df.drop('ID',axis=1,inplace=True)
smaller_df.drop('notes_for_discussion',axis=1,inplace=True)
smaller_df.drop('lives_in_Wuhan',axis=1,inplace=True)
categorical_data = [ 'age', 'sex', 'city', 'province', 'country', 'geo_resolution', 'symptoms',  'travel_history_location', 'reported_market_exposure', 'additional_information', 'chronic_disease_binary', 'chronic_disease', 'source', 'sequence_available', 'outcome', 'location', 'admin3', 'admin2', 'admin1', 'country_new', 'data_moderator_initials', 'travel_history_binary']
y = smaller_df['deceased_binary']
X = pd.DataFrame(smaller_df[categorical_data])
print(X)
cat_selection = SelectKBest(score_func=mutual_info_classif, k=10)
cat_selection.fit_transform(X,y)
for feature in range(len(cat_selection.scores_)):
    print('Feature %s: %f' % (categorical_data[feature], cat_selection.scores_[feature]))
# plot the scores
fig, ax = plt.subplots()
ax = sns.barplot(x=[categorical_data[i] for i in range(len(cat_selection.scores_))], y=cat_selection.scores_)
ax.set_title("Categorical feature selection", fontsize=10, fontweight='bold')
plt.xlabel("features")
plt.ylabel("scores")
fig.tight_layout()
plt.show()
cat_df = pd.DataFrame({
    'features':categorical_data,
    'mutual_info_classif':cat_selection.scores_
    })
cat_df = cat_df.sort_values('mutual_info_classif',ascending=False)
cat_df['rank_mutual_info_classif']=[i for i in range(len(cat_df))]
print(cat_df)

cat_selection = SelectKBest(score_func=f_classif, k=10)
cat_selection.fit_transform(X,y)
for feature in range(len(cat_selection.scores_)):
    print('Feature %s: %f' % (categorical_data[feature], cat_selection.scores_[feature]))
fig, ax = plt.subplots()
ax = sns.barplot(x=[categorical_data[i] for i in range(len(cat_selection.scores_))], y=cat_selection.scores_)
ax.set_title("Categorical feature selection", fontsize=10, fontweight='bold')
plt.xlabel("features")
plt.ylabel("scores")
fig.tight_layout()
plt.show()
cat_df['f_classif'] = cat_selection.scores_
cat_df = cat_df.sort_values('f_classif',ascending=False)
cat_df['rank_f_classif']=[i for i in range(len(cat_df))]
print(cat_df)
#Those that are ranked high in both, add rankings up and divide by two
cat_df['rank']=cat_df['rank_mutual_info_classif']+cat_df['rank_f_classif']
cat_df = cat_df.sort_values('rank',ascending=True)
print(cat_df)
best_cat_features = list(cat_df['features'].iloc[:10])
print(best_cat_features)
for col in categorical_data:
    if col not in best_cat_features:
        smaller_df.drop(col,axis=1,inplace=True)
print(smaller_df.columns)