# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 23:25:47 2020

@author: Lenovo
"""
import pickle 


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

X = pickle.load( open( "X2", "rb" ) )
X = X.to_numpy()
Y = pickle.load( open( "Y2", "rb" ) )
YY = pickle.load( open( "Y2", "rb" ) )
YY=np.reshape(YY, (YY.shape[0]))


n1 = X.shape[0]
n2 = X.shape[1]
m=int(0.8*n1)
xtrain=X[:m,:]
ytrain=YY[:m]
xtest=X[m:-1,:]
ytest=YY[m:-1]

# define 10-fold cross validation test harness
seed = 7
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []

sp=kfold.split(X, Y)
test_accuracy=[]
Y = to_categorical(Y)
for train, test in sp:
    
    model = k.models.Sequential()
    model.add(Dense(64, input_shape=(n2,),activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64,activation='relu'))
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
    
    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    score, acc = model.evaluate(X[test], Y[test])
    print(score,acc)