
# -*- coding: utf-8 -*-
"""
Created on Wed May 09 10:20:27 2018

@author: Snigdha Siddula
"""

import numpy as np
import pandas as pd
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization
from sklearn.metrics import classification_report, recall_score
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import matplotlib.image as mpimg
from keras.models import Sequential
import tensorflow as tf
from keras.layers import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv1D, MaxPooling1D
import os
os.chdir('C:/Users/Snigs/Desktop/INSOFE/day43-CUTe4')

# Data Exploration
raw_data = pd.read_csv('CUTe_data.csv',encoding='latin-1')
print('Data.Shape:',raw_data.shape)
print('\n')
print('Columns:\n',raw_data.columns)
print('\n')
print('Data Types:\n',raw_data.dtypes)
print('\n')
print(raw_data.head())

# Print the unique classes and their counts/frequencies
hate_speech = np.unique(raw_data['hate_speech'], return_counts=True) # np.unique returns a tuple with class names and counts
print('hate_speech:')
print(hate_speech[0]) # print the list of unique classes
print(hate_speech[1]) # print the list of frequencies of the above classes
print('\n')

obscene = np.unique(raw_data['obscene'],return_counts=True)
print('obscene:')
print(obscene[0])
print(obscene[1])
print('\n')

insult = np.unique(raw_data['insulting'],return_counts=True)
print('insulting:')
print(insult[0])
print(insult[1])

# Train Test Split
x = raw_data[['text']]
y = raw_data[['hate_speech','obscene','insulting']]

# Splitting data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20)
x_train.head()
y_train.head()

# Text Preprocessing
from nltk.corpus import stopwords
stop = stopwords.words('english')
x_train['text'] = x_train['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
x_train.head()

from textblob import Word
x_train['text'] = x_train['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
x_train.head()

# Converting source x data as per model intake compatibility
tokenize = Tokenizer(num_words=20000) # takes top 20000 frequenct words
tokenize.fit_on_texts(x_train.text) # taking the most frequenct words from train data

x_train_sequences = tokenize.texts_to_sequences(x_train.text)  
x_test_sequences = tokenize.texts_to_sequences(x_test.text)
x_wordindex = tokenize.word_index  # contains tokenized uniques words 

traindata_x = pad_sequences(x_train_sequences, maxlen=500) # trains on 500 word sequence
testdata_x = pad_sequences(x_test_sequences, maxlen=500)   # test on 500 word sequence

print('unique tokens:', len(x_wordindex))
print(traindata_x.shape)
print(testdata_x.shape)

# defining custom metrics as demanded.
from keras import backend as K
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


# Model Building
# Building an CNN-LSTM model
model = Sequential()
model.add(Embedding(input_dim=len(x_wordindex), 
                    input_length=500, 
                    output_dim=100))
model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(50, return_sequences=False))
model.add(BatchNormalization())
model.add(Dropout(0.5)) # to prevent overfitting
model.add(Dense(3, activation='sigmoid')) # a neuron per class in the o/p layer
model.summary()


# Mention the optimizer, Loss function and metrics to be computed
model.compile(optimizer='adam',  # 'Adam' is a variant of gradient descent technique
              loss='binary_crossentropy', # categorical_crossentropy for multi-class classification
              metrics=[recall,'accuracy']) # These metrics are computed and printed for evaluating

history = (model.fit(traindata_x, y_train, epochs=5, validation_split=0.20).history)

print(history['loss'])
print(history['acc'])
print(history['val_loss'])
print(history['val_acc'])

plt.plot(history['loss'])
plt.plot(history['val_loss'])  
plt.plot(history['recall'])
plt.plot(history['val_recall'])  

plt.title('validation recall vs validation loss')
plt.ylabel('loss/recall')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
    
model.save_weights('Weights3.h5')

# predicting on test
from numpy import array
preds = model.predict(testdata_x)
preds[preds>0.5]=1
preds[preds<0.5]=0

label1=y_test['hate_speech']
label1=np.array(label1)

label2=y_test['obscene']
label2=np.array(label2)

label3=y_test['insulting']
label3=np.array(label3)

clabel=[label1,label2,label3]
clabel=np.array(clabel)
clabel=clabel.transpose()


print(classification_report(clabel,preds))

clabel = pd.DataFrame(clabel)
clabel.head()

recall_score(clabel,preds,average="weighted")


# Weighted Recall - Calculation by hand

frac_h = 15294/31633
frac_o = 8449/31633
frac_i = 7574/31633

wt_h = 1-frac_h
wt_o = 1-frac_o
wt_i = 1-frac_i

total_wt =wt_h + wt_o + wt_i

WT_H = wt_h/total_wt
WT_O = wt_o/total_wt
WT_I = wt_i/total_wt

# Weighted Recall - update recall as per model predictions
WT_H*0.94 + WT_O*0.84 + WT_I*0.73

