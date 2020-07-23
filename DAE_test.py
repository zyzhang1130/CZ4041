# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 13:42:33 2020

@author: Lenovo
"""


from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
import tensorflow.keras as k

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.models import load_model




(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))



#adding some noise
noise_factor = 0.5
x=x_train[9]
x=np.reshape(x, (1, 28, 28, 1))
x_noisy=x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape) 
for i in range(100):
    a = x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape) 
    # x_noisy=np.stack((x_noisy, a), axis=0)
    x_noisy=np.concatenate((x_noisy, a), axis=0)

x_noisy = np.clip(x_noisy, 0., 1.)


model = load_model('denoising_autoencoder4.h5')
no_noise_img = model.predict(x_noisy)

plt.figure(figsize=(40, 4))
for i in range(10):
    # display original
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(x_noisy[i].reshape(28, 28), cmap="binary")
    
    # display reconstructed (after noise removed) image
    ax = plt.subplot(3, 20, 40 +i+ 1)
    plt.imshow(no_noise_img[i].reshape(28, 28), cmap="binary")

plt.show()