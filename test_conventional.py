# -*- coding: utf-8 -*-
"""test_conventional.ipynb

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

model.load_weights("./model/best_conventional_model.h5")

"""## confusion matrix (load testing dataset, prediciton, export)"""

if not os.path.exists("./result/conventional_train"):
  os.makedirs("./result/conventional_train")

prediction_y = model.predict([x_test, x_test, x_test, x_test, x_test, x_test])

result_matrix = np.empty((len(test_path)+1,5),dtype='U100')
result_matrix[0,0] = "images"
result_matrix[0,1] = 'Normal'
result_matrix[0,2] = "Atypical"
result_matrix[0,3] = 'Dysplasia'
result_matrix[0,4] = "CIS"

for i in range(len(test_path)):
    result_matrix[i+1,0] = str(os.path.basename(test_path[i]))
    result_matrix[i+1,1] = str(prediction_y[i][0])
    result_matrix[i+1,2] = str(prediction_y[i][1])
    result_matrix[i+1,3] = str(prediction_y[i][2])
    result_matrix[i+1,4] = str(prediction_y[i][3])


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
  elif prediction_y[i][1] > prediction_y[i][0] and prediction_y[i][1] >= prediction_y[i][2] and prediction_y[i][1] >= prediction_y[i][3]:
    predicted_class.append(1)
  elif prediction_y[i][2] > prediction_y[i][0] and prediction_y[i][2] > prediction_y[i][1] and prediction_y[i][2] >= prediction_y[i][3]:
    predicted_class.append(2)
  elif prediction_y[i][3] > prediction_y[i][0] and prediction_y[i][3] > prediction_y[i][1] and prediction_y[i][3] > prediction_y[i][2]:
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
np.savetxt("./result/conventional_train/result_row.csv" ,result_matrix, delimiter=',',fmt='%s')
#---------------------------------------------------------------------------

"""WSIs predict"""

#----------------------------------------------------------------------------
lst = pd.read_csv("./result/conventional_train/result_row.csv").values.tolist()
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
np.savetxt("./result/conventional_train/result_WSIs.csv" ,val_score ,delimiter=',',fmt='%s')
#----------------------------------------------------------------------------
print("Completed.")


#----------------------------------------------------------------------------
lst = pd.read_csv("./result/conventional_train/result_WSIs.csv" ).values.tolist()
#----------------------------------------------------------------------------
label_class = []
for i in range(len(lst)):
  if "N" in lst[i][0] :
    label_class.append("NU")
  elif "A" in lst[i][0] :
    label_class.append("AU")
  elif "D" in lst[i][0] :
    label_class.append("DP")
  elif "C" in lst[i][0] :
    label_class.append("CIS")
  else:
    print("error")

score = []
for i in range(len(lst)):
  if lst[i][1] > lst[i][2] and lst[i][1] > lst[i][3] and lst[i][1] > lst[i][4] :
    score.append("NU")
  elif lst[i][2] > lst[i][1] and lst[i][2] > lst[i][3] and lst[i][2] > lst[i][4] :
    score.append("AU")
  elif lst[i][3] > lst[i][1] and lst[i][3] > lst[i][2] and lst[i][3] > lst[i][4] :
    score.append("DP")
  elif lst[i][4] > lst[i][1] and lst[i][4] > lst[i][2] and lst[i][4] > lst[i][3] :
    score.append("CIS")
  else:
    print(str(lst[i][0]))


def print_cmx(y_true, y_pred):
    labels = ["NU", "AU","DP","CIS"]
    cmx_data_1 = confusion_matrix(y_true, y_pred, labels=labels)

    cmx_data = [[0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]]

    for j in range(4):
      for i in range(4):
        percent = cmx_data_1[j][i] / (cmx_data_1[j][0] + cmx_data_1[j][1] + cmx_data_1[j][2] + cmx_data_1[j][3])
        cmx_data[j][i] = percent

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    plt.figure(figsize = (10,7))
    sns.heatmap(df_cmx, cmap="Blues", annot=True, cbar = False, square = True)
    plt.rcParams["font.size"] = 22

    plt.title("Our model", fontsize=25)

    plt.xlabel("Predicted class", fontsize=25)
    plt.ylabel("Correct class", fontsize=25)

    plt.savefig("./result/conventional_train/result_conventional_CM_rate.tiff", format="tiff", dpi=300)
    plt.show()

print_cmx(label_class, score)
print("confusion_matrix was exported in current dirrectory")

import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
#----------------------------------------------------------------------------
lst = pd.read_csv("./result/conventional_train/result_WSIs.csv" ).values.tolist()
#----------------------------------------------------------------------------

label_class = []
for i in range(len(lst)):
  if "N" in lst[i][0] :
    label_class.append("NU")
  elif "A" in lst[i][0] :
    label_class.append("AU")
  elif "D" in lst[i][0] :
    label_class.append("DP")
  elif "C" in lst[i][0] :
    label_class.append("CIS")
  else:
    print("error")

score = []
for i in range(len(lst)):
  if lst[i][1] > lst[i][2] and lst[i][1] > lst[i][3] and lst[i][1] > lst[i][4] :
    score.append("NU")
  elif lst[i][2] > lst[i][1] and lst[i][2] > lst[i][3] and lst[i][2] > lst[i][4] :
    score.append("AU")
  elif lst[i][3] > lst[i][1] and lst[i][3] > lst[i][2] and lst[i][3] > lst[i][4] :
    score.append("DP")
  elif lst[i][4] > lst[i][1] and lst[i][4] > lst[i][2] and lst[i][4] > lst[i][3] :
    score.append("CIS")
  else:
    print(str(lst[i][0]))

def print_cmx(y_true, y_pred):
    labels = ["NU", "AU","DP","CIS"]
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    plt.figure(figsize = (10,7))
    sns.heatmap(df_cmx, cmap="Blues", annot=True, cbar = False, square = True)
    plt.rcParams["font.size"] = 19

    plt.title("Our model", fontsize=25)

    plt.xlabel("Predicted class", fontsize=25)
    plt.ylabel("Correct class", fontsize=25)

    plt.savefig("./result/conventional_train/result_conventional_CM_image.tiff", format="tiff", dpi=300)
    plt.show()

print(label_class)
print(score)

print_cmx(label_class, score)
