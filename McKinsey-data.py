
# coding: utf-8
"""
Created on Wed May 2 12:27:06 2018

@author: Snigdha Siddula
Impact of Standardization on ML-Classifier's Performance - McKinsey Dataset
"""
# import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import time
from datetime import datetime
from sklearn import preprocessing

# get current working directory
import os
os.chdir('C:/Users/Snigs/Desktop/Internship/McKinsey')

# read and explore data
DATA = pd.read_csv("train.csv")
DATA.shape
DATA.head()
DATA.dtypes

# checking for missing values
print(DATA.isnull().sum())
#print(DATA.isnull().sum().sum())

# value counts of the target
DATA['stroke'].value_counts()

print(DATA['heart_disease'].value_counts())
print(DATA['hypertension'].value_counts())
print(DATA['ever_married'].value_counts())
print(DATA['work_type'].value_counts())
print(DATA['Residence_type'].value_counts())

# Extracting target and predictors
y = DATA['stroke']
y.head()
x = DATA[['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi']]
x.head()

# Dummy variables for Categorical data
X = pd.get_dummies(x,columns=['gender','ever_married','work_type','Residence_type'])
X.head()


# WITHOUT STANDARDIZATION

# Train Test Split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y,random_state=123)
# record time -to check the impact of Standardization on model convergence time
# 1. Logistic Regression
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, accuracy_score

start_time = time.time()
logreg_train = linear_model.LogisticRegression()
lr_model = logreg_train.fit(X_train, y_train)
end_time = time.time()
print(end_time - start_time)
# predictions and accuracy
pred_lr_train = lr_model.predict(X_train)
pred_lr_test = lr_model.predict(X_test)
accuracy_score(y_test,pred_lr_test)


# 2. Naive Bayes
from sklearn.naive_bayes import GaussianNB

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
from sklearn import tree

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


# WITH STANDARDIZATION

# I. Range Scaling
from sklearn.preprocessing import MinMaxScaler # subtract min and divide by (max - min)
start_time = time.time()
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X.iloc[:,3:5])
end_time = time.time()
print(end_time - start_time)

# summarize transformed data
np.set_printoptions(precision=3)
print(rescaledX[0:5,:])

# Train Test Split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(rescaledX, y, test_size=0.3,stratify=y,random_state=123)


# 1. Logistic Regression
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


# II. Z-score
from sklearn.preprocessing import StandardScaler # subtract mean and divide by variance
scaler = StandardScaler().fit(X.iloc[:,3:5])
standardX = scaler.transform(X.iloc[:,3:5])

np.set_printoptions(precision=3)
print(standardX[0:5,:])

# Train Test Split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(standardX, y, test_size=0.3,stratify=y,random_state=123)


# 1. Logistic Regression
start_time = time.time()
logreg_train = linear_model.LogisticRegression()
lr_model = logreg_train.fit(X_train, y_train)
end_time = time.time()
print(end_time - start_time)
# predicitons
pred_lr_train = lr_model.predict(X_train)
pred_lr_test = lr_model.predict(X_test)
accuracy_score(y_test,pred_lr_test)


# 2. Naive Bayes
start_time = time.time()
nb_train = GaussianNB()
nb_model = nb_train.fit(X_train, y_train)
end_time = time.time()
print(end_time -  start_time)
# predicitons
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


# III. Normalization
from sklearn.preprocessing import Normalizer # divide by sqrt(sum of squared terms) -> L2 norm
start_time = time.time()
scaler = Normalizer().fit(X.iloc[:,3:5])
normalizedX = scaler.transform(X.iloc[:,3:5])
end_time = time.time()
print(end_time - start_time)

# summarize transformed data
np.set_printoptions(precision=3)
print(normalizedX[0:5,:])

# Train Test Split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(normalizedX, y, test_size=0.3,stratify=y,random_state=123)


# 1. Logistic Regression
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

