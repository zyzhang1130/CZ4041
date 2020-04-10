# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 16:13:25 2020

@author: Lenovo
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from collections import Counter
from sklearn.linear_model import LogisticRegression
import time

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


ord5_map = {'A': 1, 'B': 2,
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
train = pd.concat([train,pd.get_dummies(train['nom_8'], prefix='nom_8')],axis=1)
train = pd.concat([train,pd.get_dummies(train['nom_9'], prefix='nom_9')],axis=1)



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
train=data[:m,:]
test=data[m:-1,:]




def sigmoid(z):
    sigmoid_f = 1 / (1 + np.exp(-z)) #YOUR CODE HERE
    return sigmoid_f 


# construct the data matrix X

X = np.ones([n1,n2]) 
X[:,1:n2] = data[:,0:(n2-1)]
print(X.shape)
# print(X[:5,:])



# predictive function definition
def f_pred(X,w): 
    p = sigmoid(X.dot(w)) #YOUR CODE HERE
    return p


# loss function definition
def loss_logreg(y_pred,y): 
    n = len(y)
    loss = -1/n* ( y.T.dot(np.log(y_pred)) + (1-y).T.dot(np.log(1-y_pred)) ) #YOUR CODE HERE
    return loss

# run logistic regression with scikit-learn
start = time.time()
logreg_sklearn = LogisticRegression(C=1e6,random_state=0) # scikit-learn logistic regression
clf=logreg_sklearn.fit(train[:,0:(n2-1)], train[:,(n2-1)]) # learn the model parameters #YOUR CODE HERE
print('Time=',time.time() - start)


# compute loss value
w_sklearn = np.zeros([n2,1])
w_sklearn[0,0] = logreg_sklearn.intercept_
w_sklearn[1:n2,0] = logreg_sklearn.coef_
print(w_sklearn)
loss_sklearn = loss_logreg(f_pred(X,w_sklearn),data[:,(n2-1)][:,None])
print('loss sklearn=',loss_sklearn)


predict=clf.predict(test[:,0:(n2-1)])
predict_proba=clf.predict_proba(test[:,0:(n2-1)])
score=clf.score(test[:,0:(n2-1)], test[:,(n2-1)])
print('score',score)


