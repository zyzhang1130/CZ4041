# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 23:57:06 2020

@author: Lenovo
"""
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import activations
import tensorflow.keras as k
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from collections import Counter
from sklearn.linear_model import LogisticRegression
import time
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold

train=pd.read_csv('train.csv',delimiter=',') 


bin3_map = {'T': 1, 'F': -1}

bin4_map = {'Y': 1, 'N': -1}

ord1_map = {'Novice': 1, 'Contributor': 2,
               'Expert': 3, 'Master': 4, 'Grandmaster': 5}

ord2_map = {'Freezing': 1, 'Cold': 2,
               'Warm': 3, 'Hot': 4, 'Boiling Hot': 5, 'Lava Hot': 6}

ord3_map = {'a': 1, 'b': 2,
               'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8,
               'i': 9, 'j': 10, 'k': 11, 'l': 12,'m': 13, 'n': 14, 'o': 15}

ord4_map = {'A': 1, 'B': 2,
               'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8,
               'I': 9, 'J': 10, 'K': 11, 'L': 12,'M': 13, 'N': 14, 'O': 15,'P': 16, 'Q': 17,
               'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23,
               'X': 24, 'Y': 25, 'Z': 26}

train['bin_3'] = train['bin_3'].map(bin3_map)
train['bin_4'] = train['bin_4'].map(bin4_map)
train['ord_1'] = train['ord_1'].map(ord1_map)
train['ord_2'] = train['ord_2'].map(ord2_map)
train['ord_3'] = train['ord_3'].map(ord3_map)
train['ord_4'] = train['ord_4'].map(ord4_map)

train['dy_sin'] = np.sin((train['day']-1)*(2.*np.pi/7))
train['dy_cos'] = np.cos((train['day']-1)*(2.*np.pi/7))
train['mnth_sin'] = np.sin((train['month']-1)*(2.*np.pi/12))
train['mnth_cos'] = np.cos((train['month']-1)*(2.*np.pi/12))

train = pd.concat([train,pd.get_dummies(train['nom_5'], prefix='nom_5')],axis=1)
train = pd.concat([train,pd.get_dummies(train['nom_6'], prefix='nom_6')],axis=1)
train = pd.concat([train,pd.get_dummies(train['nom_7'], prefix='nom_7')],axis=1)
# train = pd.concat([train,pd.get_dummies(train['nom_8'], prefix='nom_8')],axis=1)
# train = pd.concat([train,pd.get_dummies(train['nom_9'], prefix='nom_9')],axis=1)

train = train.drop(columns="day")
train = train.drop(columns="month")
train = train.drop(columns="ord_5")
train = train.drop(columns="nom_5")
train = train.drop(columns="nom_6")
train = train.drop(columns="nom_7")
train = train.drop(columns="nom_8")
train = train.drop(columns="nom_9")
train = train.drop(columns="id")
target=train['target']
train = train.drop(columns="target")
train['target']=target


ohn=OneHotEncoder(sparse=False)
column_trans=make_column_transformer((OneHotEncoder(),['nom_0','nom_1',
'nom_2','nom_3','nom_4']),remainder='passthrough')

data=column_trans.fit_transform(train)
n1 = data.shape[0]
n2 = data.shape[1]
m=int(0.8*n1)
# train=data[:m,:]
# test=data[m:-1,:]
X = data[:,0:n2-1]
Y = data[:,n2-1]
Y = np.reshape(Y, (len(Y),1))

# X_train=train[:,0:n2-1]
# y_train=train[:,n2-1]
# y_train = np.reshape(y_train, (len(y_train),1))
# y_train = to_categorical(y_train)


# X_test=test[:,0:n2-1]
# y_test=test[:,n2-1]
# y_test = np.reshape(y_test, (len(y_test),1))
# y_test = to_categorical(y_test)

# define 10-fold cross validation test harness
seed = 7
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []

sp=kfold.split(X, Y)
test_accuracy=[]
Y = to_categorical(Y)
for train, test in sp:
    
    model = k.models.Sequential()
    model.add(Dense(1024, input_shape=(n2-1,),activation='relu'))
    model.add(Dense(512,activation='relu'))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(4,activation='relu'))
    model.add(Dense(2,activation='softmax'))
    
    
    model.compile(optimizer='Adamax',
                 loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    # patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    
    history =model.fit(X[train], Y[train],validation_data=(X[test], Y[test]),
                epochs=30,
                shuffle=True,verbose=1,callbacks=[es])
    
    # evaluate the model
    _, train_acc = model.evaluate(X[train], Y[train], verbose=1)
    _, test_acc = model.evaluate(X[test], Y[test], verbose=1)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    
    test_accuracy=max(history.history['val_acc'])
    
    # plt.figure()
    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='test')
    # plt.legend()
    # plt.show()
    # score, acc = model.evaluate(X_test, y_test)


