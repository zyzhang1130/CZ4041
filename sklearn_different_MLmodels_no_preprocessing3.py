# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 23:25:47 2020

@author: Lenovo
"""
import pickle 
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Activation
# from tensorflow.keras.models import Model
# from tensorflow.keras import activations
# import tensorflow.keras as k
# from tensorflow.keras.callbacks import EarlyStopping

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
from sklearn.model_selection import GridSearchCV 

X,YY,X_test = pickle.load( open( "data.pickle", "rb" ) )

X=X.toarray()
# X = X.to_numpy()
# Y = pickle.load( open( "Y2", "rb" ) )
# Y = to_categorical(Y)
# YY = pickle.load( open( "Y2", "rb" ) )
# YY=np.reshape(YY, (YY.shape[0]))



from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

n1 = X.shape[0]
m=int(0.8*n1)
xtrain=X[:m,:]
ytrain=YY[:m]
xtest=X[m:-1,:]
ytest=YY[m:-1]

# model = XGBClassifier(learning_rate =0.1,
#  n_estimators=1000,
#  max_depth=5,
#  min_child_weight=1,
#  gamma=0,
#  subsample=0.8,
#  colsample_bytree=0.8,
#  objective= 'binary:logistic',
#  nthread=4,
#  scale_pos_weight=1,
#  seed=27)
# model.fit(xtrain,ytrain)
# 	


# preds = model.predict(xtest)
# accuracy=accuracy_score(ytest, preds)

# param_test1 = {
#  'max_depth':range(3,10,2),
#  'min_child_weight':range(1,6,2)
# }
# gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
#  min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
#  param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# gsearch1.fit(xtrain,ytrain)
# gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_




# param_test2 = {
#  'max_depth':[4,5,6],
#  'min_child_weight':[2,3,4]
# }
# gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=5,
#  min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
#  param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# gsearch2.fit(xtrain,ytrain)
# gsearch2.best_params_, gsearch2.best_score_



# names = [ "Nearest Neighbors", 
#           "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
#           "Naive Bayes"]



# classifiers = [
#     KNeighborsClassifier(n_neighbors=7,weights='distance'),
#     # SVC(kernel="linear", C=0.025),
#     # SVC(gamma=2, C=1),
    
#     DecisionTreeClassifier(max_depth=5),
#     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#     MLPClassifier(alpha=1, max_iter=1000),
#     AdaBoostClassifier(),
#     GaussianNB()]



# s=[]

# # iterate over classifiers
# for name, clf in zip(names, classifiers):
#     print(name)
#     # clf.fit(X, Y)
#     # score = clf.score(X, Y)
#     cross_validate_score = cross_validate(clf, X, YY, cv=5, scoring="roc_auc")["test_score"].mean()
#     # print('score',score)
#     print('cross_validate_score',cross_validate_score)
#     s.append(cross_validate_score)
    
from sklearn.ensemble import BaggingClassifier
#stacking
from sklearn.ensemble import StackingClassifier
base_learners  = [
    # ('1_1',AdaBoostClassifier()),
    # ('1_2',GaussianNB()),
    ('1_1',MLPClassifier(alpha=1, max_iter=1000)),
    ('1_2',LogisticRegression(C=0.123456789, solver="lbfgs", max_iter=5000))
    # ('1_4', KNeighborsClassifier(n_neighbors=7,weights='distance')),
    # ('1_5', DecisionTreeClassifier(max_depth=9)),   
    # ('1_6', RandomForestClassifier(max_depth=12, n_estimators=13, max_features=11)),
    ]
stack_clf = StackingClassifier(estimators=base_learners,
                          final_estimator=LogisticRegression(C=0.123456789, solver="lbfgs", max_iter=5000),  
                          cv=10)
# stack_clf.fit(xtrain, ytrain)
# stack_acc=stack_clf.score(xtest, ytest)
# print('stack_acc',stack_acc)
print('1')
score=cross_validate(stack_clf, X, YY, cv=3, scoring="roc_auc")["test_score"].mean()
print(f"{score:.6f}")

#voting
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

v_clf = VotingClassifier(estimators=base_learners,voting='soft')
# v_clf.fit(X, YY)
print('2')
v_scores = cross_val_score(v_clf, X,YY, scoring="roc_auc")["test_score"].mean()
print('v_scores',v_scores)

X = X_test
X=X.toarray()
predicts=v_clf.predict(X)

    
df=pd.read_csv('submission.csv',delimiter=',') 

targets={'target':predicts}

# for i in range(n3):
#     Survivedd[Survivedd]=predicts[i]
df2 = pd.DataFrame(targets) 
df = pd.concat([df,df2],axis=1)

df.to_csv(r'submission_vote.csv', index = False)


predicts=stack_clf.predict(X)

    
df=pd.read_csv('submission.csv',delimiter=',') 

targets={'target':predicts}

# for i in range(n3):
#     Survivedd[Survivedd]=predicts[i]
df2 = pd.DataFrame(targets) 
df = pd.concat([df,df2],axis=1)

df.to_csv(r'submission_stack.csv', index = False)
