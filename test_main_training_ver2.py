"""
# -*- coding: utf-8 -*-
test_main_training.ipynb

## efficient Net モデル
"""
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")
import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
#import efficientnet.tfkeras as enet
import keras_efficientnets as enet

import os, cv2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model, load_model
#from keras.layers.core import Dense
from keras.layers.pooling import GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from keras.utils.np_utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.models import Sequential, load_model, model_from_json
from tensorflow.keras import models, optimizers, layers
from tensorflow.keras.optimizers import SGD
from keras.layers import Flatten
from sklearn.model_selection import train_test_split  
from PIL import Image 
from tensorflow.keras.preprocessing import image as images
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras import backend as K 
import numpy as np  
import glob  
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn.metrics import roc_curve
import requests

"""### １"""

#******************************************************************************
#クラスの数、クラスの名前、学習画像のサイズ    
num_classes = 2
folder = ["Low","High"]
model_number = 4

#エポック、バッチサイズ
epochs = 15
batch_size = 4

#learning_num, data_set
learnig_num = 1
data_num = 6
train_set_num = 1
day = 20201207

#validationの合計数
#val_num = 160

#augmention
augmention = False

#******************************************************************************
if model_number == 0:
    image_size = 224
elif model_number == 1:
    image_size = 240
elif model_number == 2:
    image_size = 260
elif model_number == 3:
    image_size = 300
elif model_number == 4:
    image_size = 380
elif model_number == 5:
    image_size = 456
elif model_number == 6:
    image_size = 528
elif model_number == 7:
    image_size = 600
else:
    print('set model_number 0 to 7')
    
#学習データのディレクトリ
data_path = "/media/deepstation/Transcend/data/" + "data_" + str(data_num) + "/train_" +  str(train_set_num)
print(data_path)
#モデルの名前
model_name = "/media/deepstation/Transcend/pathological_criteria_based_DL-main/model/best_model_cnn_1.h5"
print(model_name)
#.jsonの名前
json_name = "learning_" + str(learnig_num) + "_data_" + str(data_num) + "_train_set_" + str(train_set_num) + ".json"
print(json_name)
#一時保存先のpath
save_path = "/media/deepstation/Transcend/save/" + "learning_" + str(learnig_num) + "_data_" + str(data_num) + "_train_set_" + str(train_set_num) + "_" + str(day)
print(save_path)

training_path = data_path + "/training/"
validation_path = data_path + "/validation/"
testing_path = data_path + "/testing/"

from keras.backend import sigmoid

class SwishActivation(Activation):
    
    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'

def swish_act(x, beta = 1):
    return (x * sigmoid(beta * x))

from keras.utils.generic_utils import get_custom_objects
get_custom_objects().update({'swish_act': SwishActivation(swish_act)})

#base_model = model_name(weights='noisy-student',input_shape=x.shape[1:], include_top=False)
if model_number == 0:
    model = enet.EfficientNetB0(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 1:
    model = enet.EfficientNetB1(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 2:
    model = enet.EfficientNetB2(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 3:
    model = enet.EfficientNetB3(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 4:
    model = enet.EfficientNetB4(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 5:
    model = enet.EfficientNetB5(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 6:
    model = enet.EfficientNetB6(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 7:
    model = enet.EfficientNetB7(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
else:
    print('set model_number 0 to 7')

# Adding 2 fully-connected layers to B0.
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

model.summary()


# ModelCheckpoint
weights_dir='./weights/'
if os.path.exists(weights_dir)==False:os.mkdir(weights_dir)
model_checkpoint = ModelCheckpoint(
    weights_dir + "val_loss{val_loss:.3f}.hdf5",
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    period=3
)

# reduce learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, verbose=1,)
# log for TensorBoard
logging = TensorBoard(log_dir="log/")

mcp_save = ModelCheckpoint('/content/drive/My Drive/UC/model_EfficientNet/EnetB7_CIFAR10_TL.h5', save_best_only=False)

model.load_weights(model_name)

model.compile(
    optimizer = Adam(),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)


model_1_1 = Model(model.get_layer(index=0).input, model.get_layer(index=-2).output)
#model_1_2 = Model(model.get_layer(index=0).input, model.get_layer(index=-1).output)

"""### 2"""

#******************************************************************************
#クラスの数、クラスの名前、学習画像のサイズ    
num_classes = 2
folder = ["Preserve","Disturb"]
model_number = 4

#エポック、バッチサイズ
epochs = 7
batch_size = 4

#learning_num, data_set
learnig_num = 2
data_num = 6
train_set_num = 2
day = 20201207

#validationの合計数
#val_num = 160

#augmention
augmention = False

#******************************************************************************
if model_number == 0:
    image_size = 224
elif model_number == 1:
    image_size = 240
elif model_number == 2:
    image_size = 260
elif model_number == 3:
    image_size = 300
elif model_number == 4:
    image_size = 380
elif model_number == 5:
    image_size = 456
elif model_number == 6:
    image_size = 528
elif model_number == 7:
    image_size = 600
else:
    print('set model_number 0 to 7')
    
#学習データのディレクトリ
data_path = "/media/deepstation/Transcend/data/" + "data_" + str(data_num) + "/train_" +  str(train_set_num)
print(data_path)
#モデルの名前
model_name = "/media/deepstation/Transcend/pathological_criteria_based_DL-main/model/best_model_cnn_2.h5"
print(model_name)
#.jsonの名前
json_name = "learning_" + str(learnig_num) + "_data_" + str(data_num) + "_train_set_" + str(train_set_num) + ".json"
print(json_name)
#一時保存先のpath
save_path = "/media/deepstation/Transcend/save/" + "learning_" + str(learnig_num) + "_data_" + str(data_num) + "_train_set_" + str(train_set_num) + "_" + str(day)
print(save_path)

training_path = data_path + "/training/"
validation_path = data_path + "/validation/"
testing_path = data_path + "/testing/"

from keras.backend import sigmoid

class SwishActivation(Activation):
    
    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'

def swish_act(x, beta = 1):
    return (x * sigmoid(beta * x))

from keras.utils.generic_utils import get_custom_objects
get_custom_objects().update({'swish_act': SwishActivation(swish_act)})

#base_model = model_name(weights='noisy-student',input_shape=x.shape[1:], include_top=False)
if model_number == 0:
    model = enet.EfficientNetB0(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 1:
    model = enet.EfficientNetB1(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 2:
    model = enet.EfficientNetB2(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 3:
    model = enet.EfficientNetB3(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 4:
    model = enet.EfficientNetB4(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 5:
    model = enet.EfficientNetB5(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 6:
    model = enet.EfficientNetB6(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 7:
    model = enet.EfficientNetB7(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
else:
    print('set model_number 0 to 7')

# Adding 2 fully-connected layers to B0.
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

model.summary()


# ModelCheckpoint
weights_dir='./weights/'
if os.path.exists(weights_dir)==False:os.mkdir(weights_dir)
model_checkpoint = ModelCheckpoint(
    weights_dir + "val_loss{val_loss:.3f}.hdf5",
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    period=3
)

# reduce learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, verbose=1,)
# log for TensorBoard
logging = TensorBoard(log_dir="log/")

mcp_save = ModelCheckpoint('/content/drive/My Drive/UC/model_EfficientNet/EnetB7_CIFAR10_TL.h5', save_best_only=False)

model.load_weights(model_name)

model.compile(
    optimizer = Adam(),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)


model_2_1 = Model(model.get_layer(index=0).input, model.get_layer(index=-2).output)
#model_2_2 = Model(model.get_layer(index=0).input, model.get_layer(index=-1).output)

"""### 3"""

#******************************************************************************
#クラスの数、クラスの名前、学習画像のサイズ    
num_classes = 2
folder = ["Small","Large"]
model_number = 4

#エポック、バッチサイズ
epochs = 10
batch_size = 4

#learning_num, data_set
learnig_num = 1
data_num = 6
train_set_num = 3
day = 20201207

#validationの合計数
#val_num = 160

#augmention
augmention = False

#******************************************************************************
if model_number == 0:
    image_size = 224
elif model_number == 1:
    image_size = 240
elif model_number == 2:
    image_size = 260
elif model_number == 3:
    image_size = 300
elif model_number == 4:
    image_size = 380
elif model_number == 5:
    image_size = 456
elif model_number == 6:
    image_size = 528
elif model_number == 7:
    image_size = 600
else:
    print('set model_number 0 to 7')
    
#学習データのディレクトリ
data_path = "/media/deepstation/Transcend/data/" + "data_" + str(data_num) + "/train_" +  str(train_set_num)
print(data_path)
#モデルの名前
model_name = "/media/deepstation/Transcend/pathological_criteria_based_DL-main/model/best_model_cnn_3.h5"
print(model_name)
#.jsonの名前
json_name = "learning_" + str(learnig_num) + "_data_" + str(data_num) + "_train_set_" + str(train_set_num) + ".json"
print(json_name)
#一時保存先のpath
save_path = "/media/deepstation/Transcend/save/" + "learning_" + str(learnig_num) + "_data_" + str(data_num) + "_train_set_" + str(train_set_num) + "_" + str(day)
print(save_path)

training_path = data_path + "/training/"
validation_path = data_path + "/validation/"
testing_path = data_path + "/testing/"

from keras.backend import sigmoid

class SwishActivation(Activation):
    
    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'

def swish_act(x, beta = 1):
    return (x * sigmoid(beta * x))

from keras.utils.generic_utils import get_custom_objects
get_custom_objects().update({'swish_act': SwishActivation(swish_act)})

#base_model = model_name(weights='noisy-student',input_shape=x.shape[1:], include_top=False)
if model_number == 0:
    model = enet.EfficientNetB0(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 1:
    model = enet.EfficientNetB1(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 2:
    model = enet.EfficientNetB2(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 3:
    model = enet.EfficientNetB3(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 4:
    model = enet.EfficientNetB4(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 5:
    model = enet.EfficientNetB5(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 6:
    model = enet.EfficientNetB6(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 7:
    model = enet.EfficientNetB7(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
else:
    print('set model_number 0 to 7')

# Adding 2 fully-connected layers to B0.
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

model.summary()


# ModelCheckpoint
weights_dir='./weights/'
if os.path.exists(weights_dir)==False:os.mkdir(weights_dir)
model_checkpoint = ModelCheckpoint(
    weights_dir + "val_loss{val_loss:.3f}.hdf5",
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    period=3
)

# reduce learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, verbose=1,)
# log for TensorBoard
logging = TensorBoard(log_dir="log/")

mcp_save = ModelCheckpoint('/content/drive/My Drive/UC/model_EfficientNet/EnetB7_CIFAR10_TL.h5', save_best_only=False)

model.load_weights(model_name)

model.compile(
    optimizer = Adam(),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)


model_3_1 = Model(model.get_layer(index=0).input, model.get_layer(index=-2).output)
#model_3_2 = Model(model.get_layer(index=0).input, model.get_layer(index=-1).output)

"""### 4"""

#******************************************************************************
#クラスの数、クラスの名前、学習画像のサイズ    
num_classes = 2
folder = ["Various","Similar"]
model_number = 4

#エポック、バッチサイズ
epochs = 10
batch_size = 4

#learning_num, data_set
learnig_num = 1
data_num = 6
train_set_num = 4
day = 20201207

#validationの合計数
#val_num = 160

#augmention
augmention = False

#******************************************************************************
if model_number == 0:
    image_size = 224
elif model_number == 1:
    image_size = 240
elif model_number == 2:
    image_size = 260
elif model_number == 3:
    image_size = 300
elif model_number == 4:
    image_size = 380
elif model_number == 5:
    image_size = 456
elif model_number == 6:
    image_size = 528
elif model_number == 7:
    image_size = 600
else:
    print('set model_number 0 to 7')
    
#学習データのディレクトリ
data_path = "/media/deepstation/Transcend/data/" + "data_" + str(data_num) + "/train_" +  str(train_set_num)
print(data_path)
#モデルの名前
model_name = "/media/deepstation/Transcend/pathological_criteria_based_DL-main/model/best_model_cnn_4.h5"
print(model_name)
#.jsonの名前
json_name = "learning_" + str(learnig_num) + "_data_" + str(data_num) + "_train_set_" + str(train_set_num) + ".json"
print(json_name)
#一時保存先のpath
save_path = "/media/deepstation/Transcend/save/" + "learning_" + str(learnig_num) + "_data_" + str(data_num) + "_train_set_" + str(train_set_num) + "_" + str(day)
print(save_path)

training_path = data_path + "/training/"
validation_path = data_path + "/validation/"
testing_path = data_path + "/testing/"

from keras.backend import sigmoid

class SwishActivation(Activation):
    
    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'

def swish_act(x, beta = 1):
    return (x * sigmoid(beta * x))

from keras.utils.generic_utils import get_custom_objects
get_custom_objects().update({'swish_act': SwishActivation(swish_act)})

#base_model = model_name(weights='noisy-student',input_shape=x.shape[1:], include_top=False)
if model_number == 0:
    model = enet.EfficientNetB0(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 1:
    model = enet.EfficientNetB1(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 2:
    model = enet.EfficientNetB2(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 3:
    model = enet.EfficientNetB3(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 4:
    model = enet.EfficientNetB4(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 5:
    model = enet.EfficientNetB5(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 6:
    model = enet.EfficientNetB6(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 7:
    model = enet.EfficientNetB7(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
else:
    print('set model_number 0 to 7')

# Adding 2 fully-connected layers to B0.
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

model.summary()


# ModelCheckpoint
weights_dir='./weights/'
if os.path.exists(weights_dir)==False:os.mkdir(weights_dir)
model_checkpoint = ModelCheckpoint(
    weights_dir + "val_loss{val_loss:.3f}.hdf5",
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    period=3
)

# reduce learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, verbose=1,)
# log for TensorBoard
logging = TensorBoard(log_dir="log/")

mcp_save = ModelCheckpoint('/content/drive/My Drive/UC/model_EfficientNet/EnetB7_CIFAR10_TL.h5', save_best_only=False)

model.load_weights(model_name)

model.compile(
    optimizer = Adam(),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)


model_4_1 = Model(model.get_layer(index=0).input, model.get_layer(index=-2).output)
#model_4_2 = Model(model.get_layer(index=0).input, model.get_layer(index=-1).output)

"""### 5"""

#******************************************************************************
#クラスの数、クラスの名前、学習画像のサイズ    
num_classes = 2
folder = ["Negative","Positive"]
model_number = 4

#エポック、バッチサイズ
epochs = 10
batch_size = 4

#learning_num, data_set
learnig_num = 1
data_num = 6
train_set_num = 5
day = 20201207

#validationの合計数
#val_num = 160

#augmention
augmention = False

#******************************************************************************
if model_number == 0:
    image_size = 224
elif model_number == 1:
    image_size = 240
elif model_number == 2:
    image_size = 260
elif model_number == 3:
    image_size = 300
elif model_number == 4:
    image_size = 380
elif model_number == 5:
    image_size = 456
elif model_number == 6:
    image_size = 528
elif model_number == 7:
    image_size = 600
else:
    print('set model_number 0 to 7')
    
#学習データのディレクトリ
data_path = "/media/deepstation/Transcend/data/" + "data_" + str(data_num) + "/train_" +  str(train_set_num)
print(data_path)
#モデルの名前
model_name = "/media/deepstation/Transcend/pathological_criteria_based_DL-main/model/best_model_cnn_5.h5"
print(model_name)
#.jsonの名前
json_name = "learning_" + str(learnig_num) + "_data_" + str(data_num) + "_train_set_" + str(train_set_num) + ".json"
print(json_name)
#一時保存先のpath
save_path = "/media/deepstation/Transcend/save/" + "learning_" + str(learnig_num) + "_data_" + str(data_num) + "_train_set_" + str(train_set_num) + "_" + str(day)
print(save_path)

training_path = data_path + "/training/"
validation_path = data_path + "/validation/"
testing_path = data_path + "/testing/"

from keras.backend import sigmoid

class SwishActivation(Activation):
    
    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'

def swish_act(x, beta = 1):
    return (x * sigmoid(beta * x))

from keras.utils.generic_utils import get_custom_objects
get_custom_objects().update({'swish_act': SwishActivation(swish_act)})

#base_model = model_name(weights='noisy-student',input_shape=x.shape[1:], include_top=False)
if model_number == 0:
    model = enet.EfficientNetB0(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 1:
    model = enet.EfficientNetB1(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 2:
    model = enet.EfficientNetB2(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 3:
    model = enet.EfficientNetB3(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 4:
    model = enet.EfficientNetB4(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 5:
    model = enet.EfficientNetB5(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 6:
    model = enet.EfficientNetB6(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 7:
    model = enet.EfficientNetB7(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
else:
    print('set model_number 0 to 7')

# Adding 2 fully-connected layers to B0.
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

model.summary()


# ModelCheckpoint
weights_dir='./weights/'
if os.path.exists(weights_dir)==False:os.mkdir(weights_dir)
model_checkpoint = ModelCheckpoint(
    weights_dir + "val_loss{val_loss:.3f}.hdf5",
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    period=3
)

# reduce learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, verbose=1,)
# log for TensorBoard
logging = TensorBoard(log_dir="log/")

mcp_save = ModelCheckpoint('/content/drive/My Drive/UC/model_EfficientNet/EnetB7_CIFAR10_TL.h5', save_best_only=False)

model.load_weights(model_name)

model.compile(
    optimizer = Adam(),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)


model_5_1 = Model(model.get_layer(index=0).input, model.get_layer(index=-2).output)
#model_5_2 = Model(model.get_layer(index=0).input, model.get_layer(index=-1).output)

"""### 6"""

#******************************************************************************
#クラスの数、クラスの名前、学習画像のサイズ    
num_classes = 2
folder = ["Oval","Round"]
model_number = 4

#エポック、バッチサイズ
epochs = 7
batch_size = 4

#learning_num, data_set
learnig_num = 1
data_num = 6
train_set_num = 6
day = 20201207

#validationの合計数
#val_num = 160

#augmention
augmention = False

#******************************************************************************
if model_number == 0:
    image_size = 224
elif model_number == 1:
    image_size = 240
elif model_number == 2:
    image_size = 260
elif model_number == 3:
    image_size = 300
elif model_number == 4:
    image_size = 380
elif model_number == 5:
    image_size = 456
elif model_number == 6:
    image_size = 528
elif model_number == 7:
    image_size = 600
else:
    print('set model_number 0 to 7')
    
#学習データのディレクトリ
data_path = "/media/deepstation/Transcend/data/" + "data_" + str(data_num) + "/train_" +  str(train_set_num)
print(data_path)
#モデルの名前
model_name = "/media/deepstation/Transcend/pathological_criteria_based_DL-main/model/best_model_cnn_6.h5"
print(model_name)
#.jsonの名前
json_name = "learning_" + str(learnig_num) + "_data_" + str(data_num) + "_train_set_" + str(train_set_num) + ".json"
print(json_name)
#一時保存先のpath
save_path = "/media/deepstation/Transcend/save/" + "learning_" + str(learnig_num) + "_data_" + str(data_num) + "_train_set_" + str(train_set_num) + "_" + str(day)
print(save_path)

training_path = data_path + "/training/"
validation_path = data_path + "/validation/"
testing_path = data_path + "/testing/"

from keras.backend import sigmoid

class SwishActivation(Activation):
    
    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'

def swish_act(x, beta = 1):
    return (x * sigmoid(beta * x))

from keras.utils.generic_utils import get_custom_objects
get_custom_objects().update({'swish_act': SwishActivation(swish_act)})

#base_model = model_name(weights='noisy-student',input_shape=x.shape[1:], include_top=False)
if model_number == 0:
    model = enet.EfficientNetB0(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 1:
    model = enet.EfficientNetB1(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 2:
    model = enet.EfficientNetB2(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 3:
    model = enet.EfficientNetB3(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 4:
    model = enet.EfficientNetB4(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 5:
    model = enet.EfficientNetB5(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 6:
    model = enet.EfficientNetB6(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 7:
    model = enet.EfficientNetB7(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
else:
    print('set model_number 0 to 7')

# Adding 2 fully-connected layers to B0.
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

model.summary()


# ModelCheckpoint
weights_dir='./weights/'
if os.path.exists(weights_dir)==False:os.mkdir(weights_dir)
model_checkpoint = ModelCheckpoint(
    weights_dir + "val_loss{val_loss:.3f}.hdf5",
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    period=3
)

# reduce learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, verbose=1,)
# log for TensorBoard
logging = TensorBoard(log_dir="log/")

mcp_save = ModelCheckpoint('/content/drive/My Drive/UC/model_EfficientNet/EnetB7_CIFAR10_TL.h5', save_best_only=False)

model.load_weights(model_name)

model.compile(
    optimizer = Adam(),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)


model_6_1 = Model(model.get_layer(index=0).input, model.get_layer(index=-2).output)
#model_6_2 = Model(model.get_layer(index=0).input, model.get_layer(index=-1).output)

"""## 統合モデル②"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from PIL import Image
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Activation, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
import time
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from efficientnet import *
#******************************************************************************
#クラスの数、クラスの名前
num_classes = 4
folder = ["Normal","Atypical","Dysplasia","CIS"]

#学習画像のサイズ                                     
model_number = 4

#エポック、バッチサイズ
epochs = 20
batch_size = 4
save_timing = 5

#使用する学習セット
data_num = 2

train_set_num = 18
#base_set_num = 2

#validationの合計数
#val_num = 160

#学習ナンバー
learnig_num = 310
day = 20201222

#******************************************************************************

#学習データのディレクトリ
data_path = "/media/deepstation/Transcend/pathological_criteria_based_DL-main/dataset/main_train"
print(data_path)
#モデルの名前
model_naming = "learning_" + str(learnig_num) + "_data_" + str(data_num) + "_train_set_" + str(train_set_num) + "_epoch_" + str(epochs) + "_batch_" + str(batch_size) +".h5"
print(model_naming)
#.jsonの名前
json_name = "learning_" + str(learnig_num) + "_data_" + str(data_num) + "_train_set_" + str(train_set_num) + ".json"
print(json_name)
#CVSの名前
cvs_name = "learning_" + str(learnig_num) + "_data_" + str(data_num) + "_train_set_" + str(train_set_num) + ".cvs"
print(cvs_name)

#一時保存先のpath
save_path = "/media/deepstation/Transcend/save/" + "learning_" + str(learnig_num) + "_data_" + str(data_num) + "_train_set_" + str(train_set_num) + "_" + str(day)
print(save_path)

training_path = data_path + "/training/"
#base_path = "/media/deepstation/Transcend/data/data_"+ str(data_num) + "/train_" + str(base_set_num) 
validation_path = data_path+ "/validation/"
testing_path = data_path + "/testing/"

class data:
    def resize(self, x, to_color=False):
        result = []

        for i in range(len(x)):
            if to_color:
                img = cv2.cvtColor(x[i], cv2.COLOR_GRAY2RGB)
                img = cv2.resize(img,dsize=(image_size,image_size))
            else:
                img = cv2.resize(x[i],dsize=(image_size,image_size))
            result.append(img)

        return np.array(result)

    def choose_data(self, x, y, ok_label, normal_id, anomaly_id):
        x_normal, x_anomaly = [], []
        x_ref, y_ref = [], []
        j = 0
        for i in range(len(y)):
            if y[i] == normal_id:
                x_normal.append(x[i].reshape((x.shape[1:])))
            elif y[i] == 1 or y[i] ==2 or y[i] == 3 :
                x_anomaly.append(x[i].reshape((x.shape[1:])))
                #if j < len(ok_label):
                x_ref.append(x[i].reshape((x.shape[1:])))
                y_ref.append(y[i])
                #j += 1


        return np.array(x_normal), np.array(x_anomaly), np.array(x_ref), y_ref
    
    def get_data_A(self):
      img = glob.glob(training_path + "/Atypical" + "/*.png")
      img_data = np.zeros((len(img),image_size,image_size,3),dtype=np.float32)
      for i in range(len(img)):
        image = read_and_preprocess_img(img[i],size=(image_size,image_size))
        img_data[i] = image
      img_data = self.resize(img_data)
      return img_data

    def get_data_C(self):
      img = glob.glob(training_path + "/CIS"  + '/*.png')
      img_data = np.zeros((len(img),image_size,image_size,3),dtype=np.float32)
      for i in range(len(img)):
        image = read_and_preprocess_img(img[i],size=(image_size,image_size))
        img_data[i] = image
      img_data = self.resize(img_data)
      return img_data

    def get_data(self):
        #-----------------------------------------------------------------------
        

        oks = glob.glob(data_path + '/training/Normal/*.png')
        ngs = glob.glob(data_path + '/training/Atypical/*.png') + glob.glob(data_path + '/training/Dysplasia/*.png') + glob.glob(data_path + '/training/CIS/*.png')
        
        ok_data = np.zeros((len(oks),image_size,image_size,3),dtype=np.float32)
        ok_label = np.zeros(len(oks))
        
        ng_data = np.zeros((len(ngs),image_size,image_size,3),dtype=np.float32)
        ng_label = np.zeros(len(ngs))
        
        normal_path = []
        anomaly_path = []

        for i in range(len(oks)):
           print(i+1,'/',len(oks))
           image = read_and_preprocess_img(oks[i],size=(image_size,image_size))
           ok_data[i] = image
           ok_label[i] = 0
           normal_path.append(oks[i])

        
        ng_folder = ["Atypical","Dysplasia","CIS"]
        j = 0
        for index, name in enumerate(ng_folder):
          dir = training_path + name
          files = glob.glob(dir + "/*.png")    
          for i, file in enumerate(files): 
            ng_label[j] = index + 1
            j += 1
        
        for i in range(len(ngs)):
           print(i+1,'/',len(ngs))
           image = read_and_preprocess_img(ngs[i],size=(image_size,image_size))
           ng_data[i] = image
           anomaly_path.append(ngs[i])
        
        ok_data_train = ok_data
        ng_data_train = ng_data
        ok_label_train = ok_label
        ng_label_train = ng_label

        #-----------------------------------------------------------------------
        oks = glob.glob(data_path + '/testing/Normal/*.png')
        ngs = glob.glob(data_path + '/testing/Atypical/*.png') + glob.glob(data_path + '/testing/Dysplasia/*.png') + glob.glob(data_path + '/testing/CIS/*.png') 
        ok_data = np.zeros((len(oks),image_size,image_size,3),dtype=np.float32)
        ok_label = np.zeros(len(oks))
        ng_data = np.zeros((len(ngs),image_size,image_size,3),dtype=np.float32)
        ng_label = np.zeros(len(ngs))
        normal_path = []
        anomaly_path = []

        for i in range(len(oks)):
           print(i+1,'/',len(oks))
           image = read_and_preprocess_img(oks[i],size=(image_size,image_size))
           ok_data[i] = image
           ok_label[i] = 0
           normal_path.append(oks[i])
        for i in range(len(ngs)):
           print(i+1,'/',len(ngs))
           image = read_and_preprocess_img(ngs[i],size=(image_size,image_size))
           ng_data[i] = image
           ng_label[i] = 1
           anomaly_path.append(ngs[i])


        ok_data_test = ok_data
        ng_data_test = ng_data
        normal_path_test = normal_path
        anomaly_path_test = anomaly_path
        ok_label_test = ok_label
        ng_label_test = ng_label

        #--------------------------------------------------------------------





        
        x_train = np.concatenate((ok_data_train, ng_data_train),axis=0)
        x_test = np.concatenate((ok_data_test, ng_data_test),axis=0)

        y_train = np.concatenate((ok_label_train, ng_label_train),axis=0)
        y_test = np.concatenate((ok_label_test, ng_label_test),axis=0)
        

        x_train_normal, _, x_ref, y_ref = self.choose_data(x_train, y_train, ok_label_train, 0, 1)

        y_ref = to_categorical(y_ref, num_classes=4)

        x_test_normal, x_test_anomaly, _, _ = self.choose_data(x_test, y_test, ok_label_test, 0, 1)

        x_train_normal = self.resize(x_train_normal)
        x_ref = self.resize(x_ref)
        x_test_normal = self.resize(x_test_normal)
        x_test_anomaly = self.resize(x_test_anomaly)

        return x_train_normal, x_ref, y_ref, x_test_normal, x_test_anomaly, normal_path_test, anomaly_path_test


class Arcfacelayer(Layer):
    def __init__(self, output_dim, s=30, m=0.50, easy_margin=False):
        self.output_dim = output_dim
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        super(Arcfacelayer, self).__init__()

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Arcfacelayer, self).build(input_shape)
 
    def call(self, x):
        y = x[1]
        x_normalize = tf.math.l2_normalize(x[0])
        k_normalize = tf.math.l2_normalize(self.kernel)

        cos_m = K.cos(self.m)
        sin_m = K.sin(self.m)
        th = K.cos(np.pi - self.m)
        mm = K.sin(np.pi - self.m) * self.m

        cosine = K.dot(x_normalize, k_normalize)
        sine = K.sqrt(1.0 - K.square(cosine))

        phi = cosine * cos_m - sine * sin_m

        if self.easy_margin:
            phi = tf.where(cosine > 0, phi, cosine) 

        else:
            phi = tf.where(cosine > th, phi, cosine - mm) 

        output = (y * phi) + ((1.0 - y) * cosine) 
        output *= self.s

        return output

    def compute_output_shape(self, input_shape):

        return (input_shape[0][0], self.output_dim)



def no_train_arcface(x, y, classes):
    print("ArcFace training...")
    base_model=MobileNetV2(input_shape=x.shape[1:],alpha=0.5,
                           weights='imagenet',
                           include_top=False)

    c = base_model.output
    yinput = Input(shape=(classes,))
    hidden = GlobalAveragePooling2D()(c) 
    c = Arcfacelayer(classes, 30, 0.05)([hidden,yinput])
    prediction = Activation('softmax')(c)
    model = Model(inputs=[base_model.input, yinput], outputs=prediction)

#    model.compile(loss='categorical_crossentropy',
#                  optimizer=Adam(lr=0.0001, amsgrad=True),
#                  metrics=['accuracy'])

#    es_cb = keras.callbacks.EarlyStopping(monitor='loss', patience=50, verbose=0, mode='auto')

#    hist = model.fit([x, y], y, batch_size=4, epochs=epochs, verbose = 1, callbacks=[es_cb])

    return model

def get_score_arc(model, train, test):
    model = Model(model.get_layer(index=0).input, model.get_layer(index=-4).output)
    hold_vector = model.predict(train)
    predict_vector = model.predict(test)

    score = []

    for i in range(len(predict_vector)):
        cos_similarity = cosine_similarity(predict_vector[i], hold_vector)
        score.append(np.max(cos_similarity))
    return np.array(score)

def get_score_arc_2(model, train, test):
    model = Model(model.get_layer(index=0).input, model.get_layer(index=-4).output)
    hold_vector = model.predict(train)
    print(hold_vector)
    hold_vector = np.mean(hold_vector, axis=0)
    print("average")
    print(hold_vector)
    predict_vector = model.predict(test)

    score = []

    for i in range(len(predict_vector)):
        cos_similarity = cosine_similarity(predict_vector[i], hold_vector)
        score.append(np.max(cos_similarity))
    return np.array(score)

def cosine_similarity(x1, x2): 
    if x1.ndim == 1:
        x1 = x1[np.newaxis]
    if x2.ndim == 1:
        x2 = x2[np.newaxis]
    x1_norm = np.linalg.norm(x1, axis=1)
    x2_norm = np.linalg.norm(x2, axis=1)
    cosine_sim = np.dot(x1, x2.T)/(x1_norm*x2_norm+1e-10)
    return cosine_sim

def softmax(x):
    f = np.exp(x)/np.sum(np.exp(x), axis = 0, keepdims = True)
    return f

def ScoreCam(model, hold_vector, img_array, layer_name, max_N=-1):

    cls = np.argmax(model.predict([img_array,hold_vector]))
    act_map_array = Model(inputs=model.inputs, outputs=model.get_layer(index=layer_name).output).predict([img_array,hold_vector])

    input_shape = model.layers[0].output_shape[0][1:]  # get input shape
    # 1. upsampled to original input size
    act_map_resized_list = [cv2.resize(act_map_array[0,:,:,k], (input_shape[0],input_shape[1]), interpolation=cv2.INTER_LINEAR) for k in range(act_map_array.shape[3])]
    # 2. normalize the raw activation value in each activation map into [0, 1]
    act_map_normalized_list = []
    for act_map_resized in act_map_resized_list:
        if np.max(act_map_resized) - np.min(act_map_resized) != 0:
            act_map_normalized = act_map_resized / (np.max(act_map_resized) - np.min(act_map_resized))
        else:
            act_map_normalized = act_map_resized
        act_map_normalized_list.append(act_map_normalized)
    # 3. project highlighted area in the activation map to original input space by multiplying the normalized activation map
    masked_input_list = []
    for act_map_normalized in act_map_normalized_list:
        masked_input = np.copy(img_array)
        for k in range(3):
            masked_input[0,:,:,k] *= act_map_normalized
        masked_input_list.append(masked_input)
    masked_input_array = np.concatenate(masked_input_list, axis=0)
    # 4. feed masked inputs into CNN model and softmax
    hold_vector2 = np.zeros((masked_input_array.shape[0], 2))
    hold_vector2[:,1] = 1
    pred_from_masked_input_array = softmax(model.predict([masked_input_array, hold_vector2]))
    # 5. define weight as the score of target class
    weights = pred_from_masked_input_array[:,cls]
    # 6. get final class discriminative localization map as linear weighted combination of all activation maps
    cam = np.dot(act_map_array[0,:,:,:], weights)
    cam = np.maximum(0, cam)  # Passing through ReLU
    cam = cam - np.min(cam)
    cam /= np.max(cam)  # scale 0 to 1.0
    
    return cam

def superimpose(original_img_path, cam, emphasize=False):
    
    img_bgr = cv2.imread(original_img_path)

    heatmap = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))
    if emphasize:
        heatmap = sigmoid(heatmap, 50, 0.5, 1)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    hif = .8
    superimposed_img = heatmap * hif + img_bgr
    superimposed_img = np.minimum(superimposed_img, 255.0).astype(np.uint8)  # scale 0 to 255  
 
    return superimposed_img

def sigmoid(x, a, b, c):
    return c / (1 + np.exp(-a * (x-b)))

def read_and_preprocess_img(path, size=(image_size,image_size)):
    img = load_img(path, target_size=size)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

from keras.backend import sigmoid
from random import random
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, concatenate, Dense
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils.vis_utils import plot_model
import numpy as np

class SwishActivation(Activation):
    
    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'

def swish_act(x, beta = 1):
    return (x * sigmoid(beta * x))

#base_model = model_name(weights='noisy-student',input_shape=x.shape[1:], include_top=False)
if model_number == 0:
    model = enet.EfficientNetB0(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 1:
    model = enet.EfficientNetB1(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 2:
    model = enet.EfficientNetB2(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 3:
    model = enet.EfficientNetB3(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 4:
    model = enet.EfficientNetB4(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 5:
    model = enet.EfficientNetB5(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 6:
    model = enet.EfficientNetB6(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
elif model_number == 7:
    model = enet.EfficientNetB7(include_top=False, input_shape=(image_size,image_size,3), pooling='avg', weights='imagenet')
else:
    print('set model_number 0 to 7')

x1 = model_1_1.output
x2 = model_2_1.output
x3 = model_3_1.output
x4 = model_4_1.output
x5 = model_5_1.output
x6 = model_6_1.output

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

model = Model(inputs = [model_1_1.input, model_2_1.input,model_3_1.input,model_4_1.input,model_5_1.input,model_6_1.input], outputs = predictions)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
#model.load_weights("/media/deepstation/Transcend/learning/UC/model_metric_learning/Imbalanced_Image_Anomaly_Detection/learning_" + str(learnig_num) + "_data_" + str(data_num) + "_train_set_18" +"_batch_" + str(batch_size) + "_01.h5")
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
model.load_weights("/media/deepstation/Transcend/pathological_criteria_based_DL-main/model/best_main_trained_model.h5")

model.compile(
    optimizer = Adam(),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

"""## 混合配列"""

import pandas as pd
import matplotlib.pyplot as plt

def res (x):
  result = []
  for i in range(len(x)):
    img = cv2.resize(x[i],dsize=(image_size,image_size))
    result.append(img)
  return np.array(result)

oks_test = glob.glob(data_path + '/testing/Normal/*.png')
ng_1_test = glob.glob(data_path + '/testing/Atypical/*.png')
ng_2_test = glob.glob(data_path + '/testing/Dysplasia/*.png')
ng_3_test = glob.glob(data_path + '/testing/CIS/*.png')

oks_test_data = np.zeros((len(oks_test),image_size,image_size,3),dtype=np.float32)
ng_1_test_data = np.zeros((len(ng_1_test),image_size,image_size,3),dtype=np.float32)
ng_2_test_data = np.zeros((len(ng_2_test),image_size,image_size,3),dtype=np.float32)
ng_3_test_data = np.zeros((len(ng_3_test),image_size,image_size,3),dtype=np.float32)
        
path_img = []
oks_test_path = []
ng_1_test_path = []
ng_2_test_path = []
ng_3_test_path = []

for i in range(len(oks_test)):
  print(i+1,'/',len(oks_test))
  path_img.append(oks_test[i])
  image = read_and_preprocess_img(oks_test[i],size=(image_size,image_size))
  oks_test_data[i] = image
  oks_test_path.append(oks_test[i])

for i in range(len(ng_1_test)):
  print(i+1,'/',len(ng_1_test))
  path_img.append(ng_1_test[i])
  image = read_and_preprocess_img(ng_1_test[i],size=(image_size,image_size))
  ng_1_test_data[i] = image
  ng_1_test_path.append(ng_1_test[i])

for i in range(len(ng_2_test)):
  print(i+1,'/',len(ng_2_test))
  path_img.append(ng_2_test[i])
  image = read_and_preprocess_img(ng_2_test[i],size=(image_size,image_size))
  ng_2_test_data[i] = image
  ng_2_test_path.append(ng_2_test[i])

for i in range(len(ng_3_test)):
  print(i+1,'/',len(ng_3_test))
  path_img.append(ng_3_test[i])
  image = read_and_preprocess_img(ng_3_test[i],size=(image_size,image_size))
  ng_3_test_data[i] = image
  ng_3_test_path.append(ng_3_test[i])

oks_test_data = res(oks_test_data)
ng_1_test_data = res(ng_1_test_data)
ng_2_test_data = res(ng_2_test_data)
ng_3_test_data = res(ng_3_test_data)
#-------------------------------------------------------------------------------

x_test = np.vstack((oks_test_data, ng_1_test_data, ng_2_test_data, ng_3_test_data))

prediction_y = model.predict([x_test, x_test, x_test, x_test, x_test, x_test])

result_matrix = np.empty((len(path_img)+1,5),dtype='U100')
result_matrix[0,0] = "images"
result_matrix[0,1] = 'Normal'
result_matrix[0,2] = "Atypical"
result_matrix[0,3] = 'Dysplasia'
result_matrix[0,4] = "CIS"

for i in range(len(path_img)):
    result_matrix[i+1,0] = str(os.path.basename(path_img[i]))
    result_matrix[i+1,1] = str(prediction_y[i][0])
    result_matrix[i+1,2] = str(prediction_y[i][1])
    result_matrix[i+1,3] = str(prediction_y[i][2])
    result_matrix[i+1,4] = str(prediction_y[i][3])



import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def print_cmx(y_true, y_pred):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    plt.figure(figsize = (10,7))
    sns.heatmap(df_cmx, cmap="Blues", annot=True)
    plt.xlabel("Predicted_class")
    plt.ylabel("Correct_class")
    plt.show()

predicted_class = []
for i in range(len(prediction_y)):
  if prediction_y[i][0] >= prediction_y[i][1] and prediction_y[i][0] >= prediction_y[i][2] and prediction_y[i][0] >= prediction_y[i][3]:
    predicted_class.append(0)
  elif prediction_y[i][1] >= prediction_y[i][0] and prediction_y[i][1] >= prediction_y[i][2] and prediction_y[i][1] >= prediction_y[i][3]:
    predicted_class.append(1)
  elif prediction_y[i][2] >= prediction_y[i][0] and prediction_y[i][2] >= prediction_y[i][1] and prediction_y[i][2] >= prediction_y[i][3]:
    predicted_class.append(2)
  elif prediction_y[i][3] >= prediction_y[i][0] and prediction_y[i][3] >= prediction_y[i][1] and prediction_y[i][3] >= prediction_y[i][2]:
    predicted_class.append(3)
  else:
    print("error")
print(predicted_class)

n_num = glob.glob(testing_path + 'Normal/*.png')
a_num = glob.glob(testing_path + 'Atypical/*.png')
d_num = glob.glob(testing_path + 'Dysplasia/*.png')
c_num = glob.glob(testing_path + 'CIS/*.png')

label_class = []
for i in range(len(n_num)):
  label_class.append(0)
for i in range(len(a_num)):
  label_class.append(1)
for i in range(len(d_num)):
  label_class.append(2)
for i in range(len(c_num)):
  label_class.append(3)
print(label_class)

print_cmx(label_class, predicted_class)

#----------------------------------------------------------------------------
np.savetxt("result_305.csv" ,result_matrix, delimiter=',',fmt='%s')
#---------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#----------------------------------------------------------------------------
lst = pd.read_csv("result_305.csv").values.tolist()
#----------------------------------------------------------------------------

name_val = []
for i in range(len(lst)):
  lst_name = lst[i][0]
  name_val.append(lst_name[0:5])
name_val = list(set(name_val))
print(name_val)

val_score = np.empty((len(name_val)+1,5),dtype='U100')
val_score[0,0] = "name"
val_score[0,1] = 'Normal'
val_score[0,2] = "Atypical"
val_score[0,3] = "Dysplasia"
val_score[0,4] = "CIS"

num_array = []
N_array = []
A_array = []
D_array = []
C_array = []

for i in range(len(name_val)):
  num_array.append(0)
for i in range(len(name_val)):
  N_array.append(0)
for i in range(len(name_val)):
  A_array.append(0)
for i in range(len(name_val)):
  D_array.append(0)
for i in range(len(name_val)):
  C_array.append(0)


for i in range(len(lst)):
  for j in range(len(name_val)):
    if name_val[j] in lst[i][0]:
      save_index = j
  num_array[save_index] += 1
  N_array[save_index] = N_array[save_index] + lst[i][1]
  A_array[save_index] = A_array[save_index] + lst[i][2]
  D_array[save_index] = D_array[save_index] + lst[i][3]
  C_array[save_index] = C_array[save_index] + lst[i][4]


for i in range(len(name_val)):
  N = N_array[i] / num_array[i]
  A = A_array[i] / num_array[i]
  D = D_array[i] / num_array[i]
  C = C_array[i] / num_array[i]

  val_score[i+1,0] = name_val[i]
  val_score[i+1,1] = str(round(N, 10))
  val_score[i+1,2] = str(round(A, 10))
  val_score[i+1,3] = str(round(D, 10))
  val_score[i+1,4] = str(round(C, 10))
#----------------------------------------------------------------------------
np.savetxt("result_integration_4class_multi_305.csv" ,val_score ,delimiter=',',fmt='%s')
#----------------------------------------------------------------------------
print("Completed.")

"""4class"""

import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
#----------------------------------------------------------------------------
lst = pd.read_csv("result_integration_4class_multi_305.csv" ).values.tolist()
#----------------------------------------------------------------------------
label_class = []
for i in range(len(lst)):
  if "N" in lst[i][0] :
    label_class.append("0:Normal")
  elif "A" in lst[i][0] :
    label_class.append("1:Atypical U.")
  elif "D" in lst[i][0] :
    label_class.append("2:Dysplasia")
  elif "C" in lst[i][0] :
    label_class.append("3:CIS")
  else:
    print("error")

score = []
for i in range(len(lst)):
  if lst[i][1] > lst[i][2] and lst[i][1] > lst[i][3] and lst[i][1] > lst[i][4] :
    score.append("0:Normal")
  elif lst[i][2] > lst[i][1] and lst[i][2] > lst[i][3] and lst[i][2] > lst[i][4] :
    score.append("1:Atypical U.")
  elif lst[i][3] > lst[i][1] and lst[i][3] > lst[i][2] and lst[i][3] > lst[i][4] :
    score.append("2:Dysplasia")
  elif lst[i][4] > lst[i][1] and lst[i][4] > lst[i][2] and lst[i][4] > lst[i][3] :
    score.append("3:CIS")
  else:
    print(str(lst[i][0]))


def print_cmx(y_true, y_pred):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    plt.figure(figsize = (10,7))
    sns.heatmap(df_cmx, cmap="Blues", annot=True)
    plt.xlabel("Predicted_class")
    plt.ylabel("Correct_class")
    plt.savefig("result_integration_4class_multi_300.tiff", format="tiff", dpi=300)
    plt.show()

print(label_class)
print(score)

print_cmx(label_class, score)
