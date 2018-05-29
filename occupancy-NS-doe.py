
# coding: utf-8
"""
Created on Tue Apr 10 17:20:06 2018

@author: Snigdha Siddula
Impact of SMOTE and Sample on ML-Classifier's Performace - Occupancy Dataset
source: https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+
Observations: Tree-based algorithms are immune to the SMOTE sampling ratios among the binary classification
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import time
from datetime import datetime
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from collections import Counter

# change working directory
os.chdir('C:/Users/Snigs/Desktop/Internship/occupancy_data')

# read datafile and check dimensions
d_ns = pd.read_csv('data_NS.csv')
d_ns.shape
d_ns.head(5)
d_ns.dtypes

# data type conversion
d_ns['date'] = pd.DatetimeIndex(d_ns.date)
d_ns.dtypes
d_ns['Occupancy'] = d_ns.Occupancy.astype('category')
d_ns.dtypes

# checking level-counts of target
d_ns['Occupancy'].value_counts()

# extracting traget and predictors
y = d_ns['Occupancy']
y.head(5)
x = d_ns[['Temperature','Humidity','Light','CO2','HumidityRatio','year','month','day_of_week','hour_of_day']]
x.head(5)

# Train Test Split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,stratify=y,random_state=123)


# SMOTE
# Over-sampling the minority class
X_resampled, y_resampled = SMOTE().fit_sample(X_train, y_train)
print(sorted(Counter(y_resampled).items()))
type(X_resampled)

X_resampled = pd.DataFrame(X_resampled)
X_resampled.head()

type(y_resampled)

y_resampled = pd.DataFrame(y_resampled)
y_resampled.head()

# Merge resampled X and y 
train = pd.concat([X_resampled,y_resampled],axis=1)
train.columns = ['Temperature', 'Humidity', 'Light','CO2','HumidityRatio','year','month','day_of_week','hour_of_day','Occupancy']

train.head()
# where target = 0
train_0 = train[train['Occupancy']==0]
train_0.shape
# where target = 1
train_1 = train[train['Occupancy']==1]
train_1.shape

# ### SMOTE Sample Ratios - inputing target class 0 and 1 in various ratios to find the optimum ratio
train_0_samp = train_0.sample(frac=0.30,random_state=123)
train_0_samp.head()
train_1_samp = train_1.sample(frac=0.70,random_state=123)
train_1_samp.head()

# splitting train
X_train = pd.concat([train_0_samp.iloc[:,0:9],train_1_samp.iloc[:,0:9]],axis=0)
X_train.head()
y_train = pd.concat([train_0_samp['Occupancy'],train_1_samp['Occupancy']],axis=0)
y_train.head()


# CLASSIFICATION MODELLING
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, accuracy_score

#1. Logistic Regression
start_time = time.time()
logreg_train = linear_model.LogisticRegression()
lr_model = logreg_train.fit(X_train, y_train)
end_time = time.time()
print(end_time - start_time)
# predictions
pred_lr_train = lr_model.predict(X_train)
pred_lr_test = lr_model.predict(X_test)
accuracy_score(y_test,pred_lr_test)


# 2. Naive Bayes
start_time = time.time()
nb_train = GaussianNB()
nb_model = nb_train.fit(X_train, y_train)
end_time = time.time()
print(end_time -  start_time)
# predictions
pred_nb_train = nb_model.predict(X_train)
pred_nb_test = nb_model.predict(X_test)
accuracy_score(y_test,pred_nb_test)


# 3. Decision Tree
start_time = time.time()
dt = tree.DecisionTreeClassifier()
dt_model = dt.fit(X_train, y_train)
end_time = time.time()
print(end_time - start_time)
# predictions
pred_dt_test = dt_model.predict(X_test)
pred_dt_train = dt_model.predict(X_train)
accuracy_score(y_test,pred_dt_test)


# 4. SVM
from sklearn import svm
start_time = time.time()
svm = svm.SVC()
svm_model = svm.fit(X_train, y_train)
end_time = time.time()
print(end_time - start_time)
# predictions
pred_svm_test = svm_model.predict(X_test)
pred_svm_train = svm_model.predict(X_train)
accuracy_score(y_test,pred_svm_test)

