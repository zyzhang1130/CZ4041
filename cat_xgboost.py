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

X = pickle.load( open( "X2", "rb" ) )
X = X.to_numpy()
Y = pickle.load( open( "Y2", "rb" ) )
Y = to_categorical(Y)
YY = pickle.load( open( "Y2", "rb" ) )
YY=np.reshape(YY, (YY.shape[0]))



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

import xgboost as xgb
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
            
# data_dmatrix = xgb.core.DMatrix(data=data[:,0:(n2-1)],label=data[:,(n2-1)])

# params = {"objective":"multi:softmax",'colsample_bytree': 0.3,'learning_rate': 0.1,
#                 'max_depth': 5, 'alpha': 10, 'num_class':2}

# cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
#                     num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)

# xg_reg = xgb.XGBRegressor(objective ='multi:softmax', colsample_bytree = 0.3, learning_rate = 0.1,
#                 max_depth = 5, alpha = 10, n_estimators = 10,num_class=2)

# xg_reg.fit(train[:,0:(n2-1)],train[:,(n2-1)])

# preds = xg_reg.predict(test[:,0:(n2-1)])
# accuracy_score(test[:,(n2-1)], preds)

# fit model no training data
# model = XGBClassifier()
# model.fit(train[:,0:(n2-1)],train[:,(n2-1)])
# 	
# print(model)

# preds = model.predict(test[:,0:(n2-1)])
# accuracy=accuracy_score(test[:,(n2-1)], preds)



model = XGBClassifier(learning_rate =0.1,
 n_estimators=1000,
 max_depth=10,
 min_child_weight=3,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
model.fit(xtrain,ytrain)
	


preds = model.predict(xtest)
accuracy=accuracy_score(ytest, preds)
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(xtrain,ytrain)
print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)


param_test2 = {
 'max_depth':[8,9,10],
 'min_child_weight':[2,3,4]
}
gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=5,
 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2.fit(xtrain,ytrain)
print(gsearch2.best_params_, gsearch2.best_score_)


# X = pickle.load( open( "Xtest", "rb" ) )


# preds = model.predict(data)    
    
# df=pd.read_csv('submission.csv',delimiter=',') 

# Survivedd={'Survived':preds}


# df2 = pd.DataFrame(Survivedd) 
# df = pd.concat([df,df2],axis=1)

# df.to_csv(r'submission3.csv', index = False)

