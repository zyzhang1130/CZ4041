# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 09:47:54 2020

@author: Lenovo
"""


# import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs


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

X,Y=train[:,0:(n2-1)], train[:,(n2-1)]


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class SVM(nn.Module):
    """
    Linear Support Vector Machine
    -----------------------------
    This SVM is a subclass of the PyTorch nn module that
    implements the Linear  function. The  size  of  each 
    input sample is 2 and output sample  is 1.
    """
    def __init__(self):
        super().__init__()  # Call the init function of nn.Module
        self.fully_connected = nn.Linear(16199, 1)  # Implement the Linear function
        
    def forward(self, x):
        fwd = self.fully_connected(x)  # Forward pass
        return fwd
    
    
data = X  # Before feature scaling
X = (X - X.mean())/X.std()  # Feature scaling
Y[Y == 0] = -1  # Replace zeros with -1
# plt.scatter(x=X[:, 0], y=X[:, 1])  # After feature scaling
# plt.scatter(x=data[:, 0], y=data[:, 1], c='r')  # Before feature scaling


learning_rate = 0.1  # Learning rate
epoch = 10  # Number of epochs
batch_size = 1  # Batch size

# X = torch.cuda.FloatTensor(X)  # Convert X and Y to FloatTensors
Y = torch.cuda.FloatTensor(Y)

# X = torch.FloatTensor(X)  # Convert X and Y to FloatTensors
# Y = torch.FloatTensor(Y)
N = len(Y)  # Number of samples, 500

model = SVM()  # Our model
model.cuda()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # Our optimizer
model.train()  # Our model, SVM is a subclass of the nn.Module, so it inherits the train method
for epoch in range(epoch):
    perm = torch.randperm(N)  # Generate a set of random numbers of length: sample size
    sum_loss = 0  # Loss for each epoch
        
    for i in range(0, N, batch_size):
        x = X[perm[i:i + batch_size]]  # Pick random samples by iterating over random permutation
        y = Y[perm[i:i + batch_size]]  # Pick the correlating class
        
        x = torch.cuda.FloatTensor(x)  # Convert X and Y to FloatTensors
        # y = torch.from_numpy(np.array(y))
        # y = torch.cuda.FloatTensor(y)
        
        x = Variable(x)  # Convert features and classes to variables
        y = Variable(y)

        optimizer.zero_grad()  # Manually zero the gradient buffers of the optimizer
        output = model(x)  # Compute the output by doing a forward pass
        
        loss = torch.mean(torch.clamp(1 - output * y, min=0))  # hinge loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Optimize and adjust weights

        sum_loss += loss.item()  # Add the loss
        
    print("Epoch {}, Loss: {}".format(epoch, sum_loss))
    
X,Y=test[:,0:(n2-1)], test[:,(n2-1)]
data = X  # Before feature scaling
X = (X - X.mean())/X.std()  # Feature scaling
Y[Y == 0] = -1  # Replace zeros with -1
# plt.scatter(x=X[:, 0], y=X[:, 1])  # After feature scaling
# plt.scatter(x=data[:, 0], y=data[:, 1], c='r')  # Before feature scaling

X = torch.FloatTensor(X)  # Convert X and Y to FloatTensors
Y = torch.FloatTensor(Y)

x = Variable(x)  # Convert features and classes to variables
y = Variable(y)
output = model(x)
print(output) 