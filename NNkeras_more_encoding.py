# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 18:51:05 2020

@author: Lenovo
"""


# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 23:57:06 2020

@author: Lenovo
"""
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import pickle 
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

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

import pandas as pd

import category_encoders as ce

from collections import Counter
import time
import string

"""# Data"""

# Train data
# !wget -O ./train.csv --no-check-certificate "https://docs.google.com/uc?export=download&id=113cuCNN0cPPxvB2003MaZ5cmSL8O4VIK"

# Test data
# !wget -O ./test.csv --no-check-certificate "https://docs.google.com/uc?export=download&id=1aKVGUejm5PNBroaJg2r9up6SGyy58bJb"

original_train=pd.read_csv('train.csv',delimiter=',') 
original_train.head()

original_test=pd.read_csv('test.csv',delimiter=',') 
original_test.head()

"""The data essentially looks like this:

|    | Name   | dtypes | Missing | Uniques |
|----|--------|--------|---------|---------|
| 0  | id     | int64  | 0       | 300000  |
| 1  | bin_0  | int64  | 0       | 2       |
| 2  | bin_1  | int64  | 0       | 2       |
| 3  | bin_2  | int64  | 0       | 2       |
| 4  | bin_3  | object | 0       | 2       |
| 5  | bin_4  | object | 0       | 2       |
| 6  | nom_0  | object | 0       | 3       |
| 7  | nom_1  | object | 0       | 6       |
| 8  | nom_2  | object | 0       | 6       |
| 9  | nom_3  | object | 0       | 6       |
| 10 | nom_4  | object | 0       | 4       |
| 11 | nom_5  | object | 0       | 222     |
| 12 | nom_6  | object | 0       | 522     |
| 13 | nom_7  | object | 0       | 1220    |
| 14 | nom_8  | object | 0       | 2215    |
| 15 | nom_9  | object | 0       | 11981   |
| 16 | ord_0  | int64  | 0       | 3       |
| 17 | ord_1  | object | 0       | 5       |
| 18 | ord_2  | object | 0       | 6       |
| 19 | ord_3  | object | 0       | 15      |
| 20 | ord_4  | object | 0       | 26      |
| 21 | ord_5  | object | 0       | 192     |
| 22 | day    | int64  | 0       | 7       |
| 23 | month  | int64  | 0       | 12      |
| 24 | target | int64  | 0       | 2       |
"""

complete_data = original_train.append(original_test, sort=False)
num_train = len(original_train)
complete_data.head()

train = complete_data.drop(columns="target")
train.info()

"""# Binary Features

bin_0, bin_1 and bin_2 have dtypes of int64 so we don't do anything about them. However, we have to encode bin_3 and bin_4 by simply mapping them to 0's and 1's.
"""

bin3_map = {'T': 1, 'F': 0}

bin4_map = {'Y': 1, 'N': 0}

train['bin_3'] = train['bin_3'].map(bin3_map)
train['bin_4'] = train['bin_4'].map(bin4_map)

"""# Ordinal Features (with <100 uniques)

ord_0 has numerical values, so we skip it. ord_1, ord_2, ord_3 and ord_4 have significantly fewer uniques than ord_5, so we simply map them to numerical values. ord_5 is handled separately in the next section.
"""

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


train['ord_1'] = train['ord_1'].map(ord1_map)
train['ord_2'] = train['ord_2'].map(ord2_map)
train['ord_3'] = train['ord_3'].map(ord3_map)
train['ord_4'] = train['ord_4'].map(ord4_map)

"""# Ordinal Features (with >100 uniques)"""

def getASCII(letter):
    return string.ascii_letters.find(letter) + 1

train['ord_5_left'] = train['ord_5'].apply(lambda x: getASCII(x[0]))
train['ord_5_right'] = train['ord_5'].apply(lambda x: getASCII(x[1]))

train = train.drop(columns="ord_5")
train.head()

"""# Cyclical Features

day and month are cyclical in nature, so we can do the following:
"""

train['dy_sin'] = np.sin((train['day']-1)*(2.*np.pi/7))
train['dy_cos'] = np.cos((train['day']-1)*(2.*np.pi/7))
train['mnth_sin'] = np.sin((train['month']-1)*(2.*np.pi/12))
train['mnth_cos'] = np.cos((train['month']-1)*(2.*np.pi/12))

train = train.drop(columns="day")
train = train.drop(columns="month")
train = train.drop(columns="id")
train.head()

"""# Nominal Features (Low Cardinality)"""

column_trans = make_column_transformer((OneHotEncoder(sparse=False),['nom_0','nom_1','nom_2','nom_3','nom_4']),remainder='passthrough')
train_after_low_car_nom = column_trans.fit_transform(train)

pd.DataFrame(train_after_low_car_nom).head()

"""# Nominal Features (High Cardinality)"""

hashing_encoder = ce.HashingEncoder(cols=[30, 31, 32, 33, 34])
train_after_high_car_nom = hashing_encoder.fit_transform(train_after_low_car_nom)

pd.DataFrame(train_after_high_car_nom).head()

"""# Encoding Results"""

train = train_after_high_car_nom
train.info()


n1 = train.shape[0]
n2 = train.shape[1]
X_train = train[:num_train]
y_train = original_train['target'].values

X_test = train[num_train:]

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape)

# data=column_trans.fit_transform(train)
# n1 = data.shape[0]
# n2 = data.shape[1]
# m=int(0.8*n1)
# train=data[:m,:]
# test=data[m:-1,:]
X = X_train
X = X.to_numpy()
Y = y_train
Y = np.reshape(Y, (len(Y),1))
print("X: ", X.shape)
print("Y: ", Y.shape)
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
print("Y: ", Y.shape)
pickle.dump( X, open( "X", "wb" ) )
pickle.dump( Y, open( "Y", "wb" ) )
for train, test in sp:
    Y_train = to_categorical(Y[train])
    Y_test = to_categorical(Y[test])
    model = k.models.Sequential()
    model.add(Dense(64, input_shape=(n2-1,),activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(4,activation='relu'))
    model.add(Dense(2,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    
    
    model.compile(optimizer='Adamax',
                 loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    # patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    
    history =model.fit(X[train], Y_train,validation_data=(X[test], Y_test),
                epochs=30,
                shuffle=True,verbose=1,callbacks=[es])
    
    # evaluate the model
    _, train_acc = model.evaluate(X[train], Y_train, verbose=1)
    _, test_acc = model.evaluate(X[test], Y_test, verbose=1)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    
    test_accuracy=max(history.history['val_acc'])
    
    # plt.figure()
    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='test')
    # plt.legend()
    # plt.show()
    # score, acc = model.evaluate(X_test, y_test)


