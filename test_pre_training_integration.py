
"""test_pre_training_integration.ipynb
# -*- coding: utf-8 -*-
"""

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

def softmax(x):
    f = np.exp(x)/np.sum(np.exp(x), axis = 0, keepdims = True)
    return f

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
  model = Model(model.get_layer(index=0).input, model.get_layer(index=-1).output)

  return model

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

def predict_picture(img_path):
    x_test = []
    image = Image.open(img_path)
    image = image.convert("RGB")
    image = image.resize((image_size, image_size))
    data = np.asarray(image)
    x_test.append(data)
    x_test = np.array(x_test)
    x_test = x_test.astype('float32')
    x_test /= 255

    pred_1 = model_1.predict(x_test)
    pred_2 = model_2.predict(x_test)
    pred_3 = model_3.predict(x_test)
    pred_4 = model_4.predict(x_test)
    pred_5 = model_5.predict(x_test)
    pred_6 = model_6.predict(x_test)


    if pred_1[0][0] > pred_1[0][1] :
        print("cellularity")
        print(">>>Low")
        print("probability")
        print(pred_1[0][1]*100)
    else :
        print("cellularity")
        print(">>>High")
        print("probability")
        print(pred_1[0][1]*100)

    if pred_2[0][0] > pred_2[0][1] :
        print("Pol")
        print(">>>Preseve")
        print("probability")
        print(pred_2[0][1]*100)
    else :
        print("Pol")
        print(">>>Disturb")
        print("probability")
        print(pred_2[0][1]*100)

    if pred_3[0][0] > pred_3[0][1] :
        print("size")
        print(">>>small")
        print("probability")
        print(pred_3[0][1]*100)
    else :
        print("size")
        print(">>>large")
        print("probability")
        print(pred_3[0][1]*100)

    if pred_4[0][0] > pred_4[0][1] :
        print("size_variety")
        print(">>>similar")
        print("probability")
        print(pred_4[0][1]*100)
    else :
        print("size_variety")
        print(">>>various")
        print("probability")
        print(pred_4[0][1]*100)

    if pred_5[0][0] > pred_5[0][1] :
        print("shape")
        print(">>>oval")
        print("probability")
        print(pred_5[0][1]*100)
    else :
        print("shape")
        print(">>>round")
        print("probability")
        print(pred_5[0][1]*100)

    if pred_6[0][0] > pred_6[0][1] :
        print("chromatin")
        print(">>>Negative")
        print("probability")
        print(pred_6[0][1]*100)
    else :
        print("chromatin")
        print(">>>Positive")
        print("probability")
        print(pred_6[0][1]*100)

    img_show = cv2.imread(img_path)
    plt.imshow(img_show)

    return pred_1[0][1], pred_2[0][1], pred_3[0][1], pred_4[0][1], pred_5[0][1], pred_6[0][1]

pred_1, pred_2, pred_3, pred_4, pred_5, pred_6 = predict_picture(image_path)
