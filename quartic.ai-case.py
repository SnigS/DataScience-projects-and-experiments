
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import confusion_matrix,recall_score, precision_score, accuracy_score, roc_auc_score
import os


# In[2]:


os.chdir("C:/Users/Snigs/Desktop/DS projects/ds_data_big/ds_data")


# ### Load data_train

# In[3]:


# read the train datafile
TRAIN = pd.read_csv("data_train.csv")
print("Train Dimensions:",TRAIN.shape)
print("Train Data:\n", TRAIN.head())
print("Data Types:\n",TRAIN.dtypes)


# In[4]:


# checking the class balance of the taget ---> clear imbalance indicated
print("Target Table:\n",TRAIN['target'].value_counts())

# isolate Target
target = TRAIN['target']
print("Target:\n",target.value_counts())


# In[5]:


# checking for missing values
print("Total Missing Values:",TRAIN.isnull().sum().sum())
print("Missing Values per Feature:\n", TRAIN.isnull().sum())


# In[6]:


# 69% of cat6 and 44.7% of cat8 values are missing ... hence the columns can be dropped
# id column has to be dropped and since we have isolated the target it can also be dropped
train = TRAIN.drop(['id','cat6','cat8','target'],axis=1)
train.shape


# ### Load data_test

# In[7]:


# read the test datafile
TEST = pd.read_csv("data_test.csv")
print("Test Dimensions:", TEST.shape)
print("Test Data:\n", TEST.head())
print("Data Types:\n",TEST.dtypes)


# In[8]:


# checking for missing values
print("Total Missing Values:",TEST.isnull().sum().sum())
print("Missing Values per Feature:\n", TEST.isnull().sum())


# In[9]:


# 69% of cat6 and 44.8% of cat8 values are missing ... hence the columns can be dropped
# id column has to be dropped too
test = TEST.drop(['id','cat6','cat8'],axis=1)
test.shape


# ### TRAIN + TEST

# In[10]:


# combine train and test by column for the ease of imputing missing values...and then can be separated by the row ids
TT = pd.concat([train,test])
print("Train-Test Dimensions:", TT.shape)
print("Train-Test Data:\n", TT.head())
print("Train-Test Data Types:\n",TT.dtypes)


# In[11]:


# checking for missing values
print("Total Missing Values:",TT.isnull().sum().sum())
print("Missing Values per Feature:\n",TT.isnull().sum())


# #### Dealing with Missing Values

# #from fancyimpute import KNN
# #tt_noNA = KNN(k=3).complete(tt)

# In[12]:


# imputing missing values
TT_noNA = TT.apply(lambda x:x.fillna(x.value_counts().index[0]))


# In[13]:


# checking for the total missing values post imputation ... should be 0
TT_noNA.isnull().sum().sum()


# ### Splitting tt (Train-Test) to their original dimensions

# In[14]:


# train data without missing values
train_noNA = TT_noNA.iloc[0:596000,:]
train_noNA.shape


# In[15]:


# test data without missing values
test_noNA = TT_noNA.iloc[596000:1488816,:]
test_noNA.shape


# ### SMOTE - oversampling the minority class to make up for the class imbalanced target

# In[16]:


from imblearn.over_sampling import SMOTE
from collections import Counter

X,Y = SMOTE().fit_sample(train_noNA,target)
print(sorted(Counter(Y).items()))


# ### Train-Validation Split

# In[17]:


from sklearn.cross_validation import train_test_split

x_train, x_val, y_train, y_val = train_test_split(X,Y, test_size=0.3,random_state=20)


# ### K-Fold Cross Validation

# #kf = KFold(n_splits=5, random_state=None, shuffle=False) 
# 
# #for train_index, test_index in kf.split(X):
#     print("Train:", train_index, "Validation:",test_index)
#     x_train, x_val = X[train_index], X[test_index] 
#     y_train, y_val = Y[train_index], Y[test_index]

# ### Classification Modelling

# In[18]:


#1. Logistic Regression

logreg_train = linear_model.LogisticRegression(penalty='l2',C=1.0)
lr_model = logreg_train.fit(x_train, y_train)

pred_lr_train = lr_model.predict(x_train)
pred_lr_val = lr_model.predict(x_val)

# observe both train and test accuracy to understand model performance i.e underfit or overfit
print('train accuracy:',accuracy_score(y_train,pred_lr_train))
print('validation accuracy:',accuracy_score(y_val, pred_lr_val))

print('Precision:',precision_score(y_val,pred_lr_val))

print('AUC-ROC:',roc_auc_score(y_val,pred_lr_val))


# In[19]:


#2. Naive Bayes

nb_train = GaussianNB()
nb_model = nb_train.fit(x_train, y_train)

pred_nb_train = nb_model.predict(x_train)
pred_nb_val = nb_model.predict(x_val)

# observe both train and test accuracy to understand model performance i.e underfit or overfit
print('train accuracy:',accuracy_score(y_train,pred_nb_train))
print('test accuracy:',accuracy_score(y_val,pred_nb_val))

print('Precision:',precision_score(y_val,pred_nb_val))

print('AUC-ROC:',roc_auc_score(y_val,pred_nb_val))


# In[20]:


#3. Decision Tree

dt = tree.DecisionTreeClassifier()
dt_model = dt.fit(x_train, y_train)

# observe both train and test accuracy to understand model performance i.e underfit or overfit
pred_dt_train = dt_model.predict(x_train)
pred_dt_val = dt_model.predict(x_val)

print('train accuracy:',accuracy_score(y_train,pred_dt_train))
print('test accuracy:',accuracy_score(y_val,pred_dt_val))

print('Precision:',precision_score(y_val,pred_dt_val))

print('AUC-ROC:',roc_auc_score(y_val,pred_dt_val))


# In[21]:


#4. Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf_model = rf.fit(x_train,y_train)

pred_rf_train = rf_model.predict(x_train)
pred_rf_val = rf_model.predict(x_val)

# observe both train and test accuracy to understand model performance i.e underfit or overfit
print('train accuracy:',accuracy_score(y_train,pred_rf_train))
print('test acuracy:',accuracy_score(y_val,pred_rf_val))

print('Precision:',precision_score(y_val,pred_rf_val))

print('AUC-ROC:',roc_auc_score(y_val,pred_rf_val))


# ### Predicitons on Test dataset

# In[22]:


target = dt_model.predict(test_noNA)
target = pd.DataFrame(target, columns =['target'])


# In[23]:


# getting the id column from test data set
id = TEST['id']
id = pd.DataFrame(id)

# write to csv
test_target = pd.concat(objs=[id, target], axis=1)
test_target.to_csv('test_target.csv',index=False)


# ### Positives(1s) predicted on the test from the above Classification Models:
# The observed positives(=1) in the data_train is 3.64%
# 1. logistic regression ~ 352025
# 2. Naive Bayes ~ 375976
# 3. Decision Tree  ~ 45333 (~ 5.07%)
# 4. Random Forest ~ 79
# 
# From the EDA of the target given in the train data set, I choose to go with the predictions of the Decision Tree model as it seems to match the class imbalance proportion.
