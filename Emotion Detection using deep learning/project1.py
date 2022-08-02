# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 10:33:26 2022

@author: SUN RISE
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 00:53:50 2022

@author: SUN RISE
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import datasets,layers,models
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.utils import to_categorical, plot_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

model = tf.keras.models.load_model('emotion_detection.h5',compile=False) #same file path
model.summary()

import os
import cv2
from keras.preprocessing import image
import imageio as iio
img=iio.imread('sad.jpg')
im=cv2.imread('sad.jpg')
cv2.imshow('img',im)
frame_grey=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
print('Resized Dimensions : ',frame_grey.shape)
datu=np.array(frame_grey)
normu_dat=datu
normu_dat=datu/255


