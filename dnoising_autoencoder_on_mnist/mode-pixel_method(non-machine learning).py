# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 21:17:27 2020

@author: Lenovo
"""

from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

n=30
#adding some noise
noise_factor = 1.2
x=x_train[7]
x=np.reshape(x, (1, 28, 28, 1))
x_noisy=x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape) 
for i in range(n):
    a = x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape) 
    # x_noisy=np.stack((x_noisy, a), axis=0)
    x_noisy=np.concatenate((x_noisy, a), axis=0)

x_noisy = np.clip(x_noisy, 0., 1.)

#isplaying images with noise
plt.figure(figsize=(20, 2))
for i in range(1,10):
    ax = plt.subplot(1, 10, i)
    plt.imshow(x_noisy[i].reshape(28, 28), cmap="binary")
plt.show()

d=np.zeros((28,28))
for j in range(28):
    for k in range(28):
        a=[]
        for i in range(n):
            a.append(x_noisy[i,j,k,0])
        data = Counter(a)
        d[j,k]=data.most_common(1)[0][0]
        
# display original
plt.imshow(d, cmap="binary")

plt.show()
            