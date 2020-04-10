# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 22:46:12 2020

@author: Lenovo
"""

import pickle 

from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Activation
from tensorflow.keras import regularizers

X = pickle.load( open( "X", "rb" ) )
X = X.to_numpy()
Y = pickle.load( open( "Y", "rb" ) )

model = Sequential()
model.add(Dense(30000))
model.add(Activation('relu'))
model.add(Dense(1, kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('linear'))
model.compile(loss='hinge',
              optimizer='adadelta',
              metrics=['accuracy'])
model.fit(X, Y)


