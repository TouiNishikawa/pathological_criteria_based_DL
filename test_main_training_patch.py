# -*- coding: utf-8 -*-
"""test_main_training.ipynb

"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /media/deepstation/Transcend/pathological_criteria_based_DL-main

from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Flatten, concatenate
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import keras_efficientnets as enet

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import glob
import numpy as np
import os, cv2
from PIL import Image
import tensorflow as tf
from keras.backend import sigmoid
from keras.utils.generic_utils import get_custom_objects
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.keras.layers import Input, GlobalAveragePooling2D
from sklearn import metrics
from efficientnet import *

import pandas as pd
from random import random
from tensorflow.python.keras.utils.vis_utils import plot_model

import seaborn as sns
from sklearn.metrics import confusion_matrix
import sys
image_path_argv = sys.argv
image_path = image_path_argv[1]

image_size = 380

"""## Load pre-training models"""

class SwishActivation(Activation):

    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'

def swish_act(x, beta = 1):
    return (x * sigmoid(beta * x))

def res (x):
  result = []
  for i in range(len(x)):
    img = cv2.resize(x[i],dsize=(image_size,image_size))
    result.append(img)
  return np.array(result)

def cnn_generate (model_name):
  num_classes = 2
  image_size = 380
  get_custom_objects().update({'swish_act': SwishActivation(swish_act)})

  model = enet.EfficientNetB4(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
  x = model.output
  x = BatchNormalization()(x)
  x = Dropout(0.7)(x)
  x = Dense(512)(x)
  x = BatchNormalization()(x)
  x = Activation(swish_act)(x)
  x = Dropout(0.5)(x)
  x = Dense(128)(x)
  x = BatchNormalization()(x)
  x = Activation(swish_act)(x)
  # Output layer
  predictions = Dense(num_classes, activation="softmax")(x)
  model = Model(inputs = model.input, outputs = predictions)

  model.load_weights(model_name)
  model = Model(model.get_layer(index=0).input, model.get_layer(index=-2).output)

  return model

def softmax(x):
    f = np.exp(x)/np.sum(np.exp(x), axis = 0, keepdims = True)
    return f

model_1 = cnn_generate("./model/best_model_cnn_1.h5")
model_1.summary()
model_2 = cnn_generate("./model/best_model_cnn_2.h5")
model_2.summary()
model_3 = cnn_generate("./model/best_model_cnn_3.h5")
model_3.summary()
model_4 = cnn_generate("./model/best_model_cnn_4.h5")
model_4.summary()
model_5 = cnn_generate("./model/best_model_cnn_5.h5")
model_5.summary()
model_6 = cnn_generate("./model/best_model_cnn_6.h5")
model_6.summary()

"""## Main training"""

#******************************************************************************
num_classes = 4
folder = ["Normal","Atypical","Dysplasia","CIS"]
epochs = 15
batch_size = 4
#******************************************************************************
data_path = "./dataset/main_train"
print(data_path)

training_path = data_path + "/training/"
validation_path = data_path+ "/validation/"
testing_path = data_path + "/testing/"

x1 = model_1.output
x2 = model_2.output
x3 = model_3.output
x4 = model_4.output
x5 = model_5.output
x6 = model_6.output

combined = concatenate([x1, x2, x3, x4, x5, x6])


# 密結合
x = Dense(512)(combined)
x = BatchNormalization()(x)
x = Activation(swish_act)(x)
x = Dropout(0.5)(x)

x = Dense(128)(x)
x = BatchNormalization()(x)
x = Activation(swish_act)(x)


# Output layer
predictions = Dense(num_classes, activation="softmax")(x)

model = Model(inputs = [model_1.input, model_2.input,model_3.input,model_4.input,model_5.input,model_6.input], outputs = predictions)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()


model.load_weights("./model/best_main_trained_model.h5")



image = Image.open(image_path)
image = image.convert("RGB")
image = image.resize((image_size, image_size))
data = np.asarray(image)
x_test.append(data)
x_test = np.array(x_test)
x_test = x_test.astype('float32')
x_test /= 255

prediction_y = model.predict([x_test, x_test, x_test, x_test, x_test, x_test])

print("The probability of NU:")
print(prediction_y[0])
print("The probability of AU:")
print(prediction_y[1])
print("The probability of DP:")
print(prediction_y[2])
print("The probability of CIS:")
print(prediction_y[3])
