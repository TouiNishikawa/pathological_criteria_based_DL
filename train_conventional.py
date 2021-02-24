# -*- coding: utf-8 -*-
"""train_conventional.ipynb
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

def cnn_generate ():
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

  model = Model(model.get_layer(index=0).input, model.get_layer(index=-2).output)

  return model

def softmax(x):
    f = np.exp(x)/np.sum(np.exp(x), axis = 0, keepdims = True)
    return f

model_1 = cnn_generate()
model_1.summary()
model_2 = cnn_generate()
model_2.summary()
model_3 = cnn_generate()
model_3.summary()
model_4 = cnn_generate()
model_4.summary()
model_5 = cnn_generate()
model_5.summary()
model_6 = cnn_generate()
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

x_train = []
y_train = []
train_path = []
x_val = []
y_val = []
val_path = []
x_test = []
y_test = []
test_path = []


for index, name in enumerate(folder):
    dir = training_path + name
    files = glob.glob(dir + "/*.png")
    for i, file in enumerate(files):
        train_path.append(file)
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        x_train.append(data)
        y_train.append(index)
        print("training_" + str(name) + ":　"+ str(i))

x_train = np.array(x_train)
y_train = np.array(y_train)

for index, name in enumerate(folder):
    dir = validation_path + name
    files = glob.glob(dir + "/*.png")
    for i, file in enumerate(files):
        val_path.append(file)
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        x_val.append(data)
        y_val.append(index)
        print("validation_" + str(name) + ":　"+ str(i))

x_val = np.array(x_val)
y_val = np.array(y_val)

for index, name in enumerate(folder):
    dir = testing_path + name
    files = glob.glob(dir + "/*.png")
    for i, file in enumerate(files):
        test_path.append(file)
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        x_test.append(data)
        y_test.append(index)
        print("testing_" + str(name) + ":　"+ str(i))

x_test = np.array(x_test)
y_test = np.array(y_test)


x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_val /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(x_train.shape[0], 'train samples')
print(x_val.shape[0], 'validation samples')
print(x_test.shape[0], 'test samples')

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

print("training...")

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001, amsgrad=True),
              metrics=['accuracy'])

if not os.path.exists("./model/conventional_models"):
  os.makedirs("./model/conventional_models")

# checkpointの設定
checkpoint = ModelCheckpoint(
                    filepath="./model/conventional_models/conventional_model" + "_batch_" + str(batch_size) + "_{epoch:02d}.h5",
                    monitor='val_loss',
                    save_best_only=False,
                    period=1,
                )

es_cb = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2, verbose=1, mode='auto')

# reduce learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=1, verbose=1,)

hist = model.fit([x_train, x_train, x_train, x_train, x_train, x_train], y_train, batch_size=batch_size, epochs=epochs, verbose = 1, callbacks=[es_cb, checkpoint, reduce_lr], validation_data=([x_val,x_val,x_val,x_val,x_val,x_val], y_val))

if not os.path.exists("./log/conventiobnal_train"):
  os.makedirs("./log/conventiobnal_train")

loss_history = hist.history["val_loss"]
np_loss_history = np.array(loss_history)
np.savetxt("./log/conventiobnal_train/loss_history.txt", np_loss_history, delimiter=",")

acc_history = hist.history["val_accuracy"]
np_acc_history = np.array(acc_history)
np.savetxt("./log/conventiobnal_train/acc_history.txt", np_acc_history, delimiter=",")

import shutil

max_index = np.argmax(acc_history)

best_path = "./model/conventional_models/conventional_model" + "_batch_" + str(batch_size) + "_" + "%02d"%int(max_index+1)  + ".h5"
shutil.copyfile(best_path , "./model/best_conventional_model.h5")
