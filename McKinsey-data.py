
# coding: utf-8
"""
Created on Wed May 2 12:27:06 2018

@author: Snigdha Siddula
Impact of Standardization on ML-Classifier's Performance - McKinsey Dataset
"""
# In[123]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import time
from datetime import datetime
from sklearn import preprocessing


# In[124]:


# get current working directory
import os
os.chdir('C:/Users/Snigs/Desktop/Internship/McKinsey')


# In[125]:


DATA = pd.read_csv("train.csv")
DATA.shape


# In[126]:


DATA.head()


# In[127]:


DATA.dtypes


# In[128]:


# checking for missing values
print(DATA.isnull().sum())
#print(DATA.isnull().sum().sum())


# In[129]:


# value counts of the target
DATA['stroke'].value_counts()


# In[130]:


print(DATA['heart_disease'].value_counts())
print(DATA['hypertension'].value_counts())
print(DATA['ever_married'].value_counts())
print(DATA['work_type'].value_counts())
print(DATA['Residence_type'].value_counts())


# In[131]:


# Extracting target and predictors
y = DATA['stroke']
y.head()


# In[132]:


x = DATA[['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi']]
x.head()


# In[133]:


# Dummy variables for Categorical data
X = pd.get_dummies(x,columns=['gender','ever_married','work_type','Residence_type'])
X.head()


# ### WITHOUT STANDARDIZATION

# In[134]:


# Train Test Split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y,random_state=123)


# ### 1. Logistic Regression

# In[135]:


from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, accuracy_score


# In[136]:


start_time = time.time()
logreg_train = linear_model.LogisticRegression()
lr_model = logreg_train.fit(X_train, y_train)
end_time = time.time()
print(end_time - start_time)


# In[137]:


pred_lr_train = lr_model.predict(X_train)
pred_lr_test = lr_model.predict(X_test)
accuracy_score(y_test,pred_lr_test)


# ### 2. Naive Bayes

# In[138]:


from sklearn.naive_bayes import GaussianNB


# In[139]:


start_time = time.time()
nb_train = GaussianNB()
nb_model = nb_train.fit(X_train, y_train)
end_time = time.time()
print(end_time -  start_time)


# In[140]:


pred_nb_train = nb_model.predict(X_train)
pred_nb_test = nb_model.predict(X_test)
accuracy_score(y_test,pred_nb_test)


# ### 3. Decision Tree

# In[141]:


from sklearn import tree


# In[142]:


start_time = time.time()
dt = tree.DecisionTreeClassifier()
dt_model = dt.fit(X_train, y_train)
end_time = time.time()
print(end_time - start_time)


# In[143]:


pred_dt_test = dt_model.predict(X_test)
pred_dt_train = dt_model.predict(X_train)
accuracy_score(y_test,pred_dt_test)


# ### 4. SVM

# In[144]:


from sklearn import svm
start_time = time.time()
svm = svm.SVC()
svm_model = svm.fit(X_train, y_train)
end_time = time.time()
print(end_time - start_time)


# In[145]:


pred_svm_test = svm_model.predict(X_test)
pred_svm_train = svm_model.predict(X_train)
accuracy_score(y_test,pred_svm_test)


# ### WITH STANDARDIZATION

# ### Range Scaling

# In[146]:


from sklearn.preprocessing import MinMaxScaler # subtract min and divide by (max - min)
start_time = time.time()
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X.iloc[:,3:5])
end_time = time.time()
print(end_time - start_time)


# In[147]:


# summarize transformed data
np.set_printoptions(precision=3)
print(rescaledX[0:5,:])


# In[148]:


# Train Test Split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(rescaledX, y, test_size=0.3,stratify=y,random_state=123)


# ### 1. Logistic Regression

# In[149]:


start_time = time.time()
logreg_train = linear_model.LogisticRegression()
lr_model = logreg_train.fit(X_train, y_train)
end_time = time.time()
print(end_time - start_time)


# In[150]:


pred_lr_train = lr_model.predict(X_train)
pred_lr_test = lr_model.predict(X_test)
accuracy_score(y_test,pred_lr_test)


# ### 2. Naive Bayes

# In[151]:


start_time = time.time()
nb_train = GaussianNB()
nb_model = nb_train.fit(X_train, y_train)
end_time = time.time()
print(end_time -  start_time)


# In[152]:


pred_nb_train = nb_model.predict(X_train)
pred_nb_test = nb_model.predict(X_test)
accuracy_score(y_test,pred_nb_test)


# ### 3. Decision Tree

# In[153]:


start_time = time.time()
dt = tree.DecisionTreeClassifier()
dt_model = dt.fit(X_train, y_train)
end_time = time.time()
print(end_time - start_time)


# In[154]:


pred_dt_test = dt_model.predict(X_test)
pred_dt_train = dt_model.predict(X_train)
accuracy_score(y_test,pred_dt_test)


# ### 4. SVM

# In[155]:


from sklearn import svm
start_time = time.time()
svm = svm.SVC()
svm_model = svm.fit(X_train, y_train)
end_time = time.time()
print(end_time - start_time)


# In[156]:


pred_svm_test = svm_model.predict(X_test)
pred_svm_train = svm_model.predict(X_train)
accuracy_score(y_test,pred_svm_test)


# ### Z-score

# In[157]:


from sklearn.preprocessing import StandardScaler # subtract mean and divide by variance
scaler = StandardScaler().fit(X.iloc[:,3:5])
standardX = scaler.transform(X.iloc[:,3:5])


# In[158]:


np.set_printoptions(precision=3)
print(standardX[0:5,:])


# In[159]:


# Train Test Split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(standardX, y, test_size=0.3,stratify=y,random_state=123)


# ### 1. Logistic Regression

# In[160]:


start_time = time.time()
logreg_train = linear_model.LogisticRegression()
lr_model = logreg_train.fit(X_train, y_train)
end_time = time.time()
print(end_time - start_time)


# In[161]:


pred_lr_train = lr_model.predict(X_train)
pred_lr_test = lr_model.predict(X_test)
accuracy_score(y_test,pred_lr_test)


# ### 2. Naive Bayes

# In[162]:


start_time = time.time()
nb_train = GaussianNB()
nb_model = nb_train.fit(X_train, y_train)
end_time = time.time()
print(end_time -  start_time)


# In[163]:


pred_nb_train = nb_model.predict(X_train)
pred_nb_test = nb_model.predict(X_test)
accuracy_score(y_test,pred_nb_test)


# ### 3. Decision Tree

# In[164]:


start_time = time.time()
dt = tree.DecisionTreeClassifier()
dt_model = dt.fit(X_train, y_train)
end_time = time.time()
print(end_time - start_time)


# In[165]:


pred_dt_test = dt_model.predict(X_test)
pred_dt_train = dt_model.predict(X_train)
accuracy_score(y_test,pred_dt_test)


# ### 4. SVM

# In[166]:


from sklearn import svm
start_time = time.time()
svm = svm.SVC()
svm_model = svm.fit(X_train, y_train)
end_time = time.time()
print(end_time - start_time)


# In[167]:


pred_svm_test = svm_model.predict(X_test)
pred_svm_train = svm_model.predict(X_train)
accuracy_score(y_test,pred_svm_test)


# ### Normalization

# In[168]:


from sklearn.preprocessing import Normalizer # divide by sqrt(sum of squared terms) -> L2 norm
start_time = time.time()
scaler = Normalizer().fit(X.iloc[:,3:5])
normalizedX = scaler.transform(X.iloc[:,3:5])
end_time = time.time()
print(end_time - start_time)


# In[169]:


# summarize transformed data
np.set_printoptions(precision=3)
print(normalizedX[0:5,:])


# In[170]:


# Train Test Split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(normalizedX, y, test_size=0.3,stratify=y,random_state=123)


# ### 1. Logistic Regression

# In[171]:


start_time = time.time()
logreg_train = linear_model.LogisticRegression()
lr_model = logreg_train.fit(X_train, y_train)
end_time = time.time()
print(end_time - start_time)


# In[172]:


pred_lr_train = lr_model.predict(X_train)
pred_lr_test = lr_model.predict(X_test)
accuracy_score(y_test,pred_lr_test)


# ### 2. Naive Bayes

# In[173]:


start_time = time.time()
nb_train = GaussianNB()
nb_model = nb_train.fit(X_train, y_train)
end_time = time.time()
print(end_time -  start_time)


# In[174]:


pred_nb_train = nb_model.predict(X_train)
pred_nb_test = nb_model.predict(X_test)
accuracy_score(y_test,pred_nb_test)


# ### 3. Decision Tree

# In[175]:


start_time = time.time()
dt = tree.DecisionTreeClassifier()
dt_model = dt.fit(X_train, y_train)
end_time = time.time()
print(end_time - start_time)


# In[176]:


pred_dt_test = dt_model.predict(X_test)
pred_dt_train = dt_model.predict(X_train)
accuracy_score(y_test,pred_dt_test)


# ### 4. SVM

# In[177]:


from sklearn import svm
start_time = time.time()
svm = svm.SVC()
svm_model = svm.fit(X_train, y_train)
end_time = time.time()
print(end_time - start_time)


# In[178]:


pred_svm_test = svm_model.predict(X_test)
pred_svm_train = svm_model.predict(X_train)
accuracy_score(y_test,pred_svm_test)

