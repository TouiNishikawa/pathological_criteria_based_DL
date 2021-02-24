# -*- coding: utf-8 -*-
"""train_cnn1.py

## 1. Set up envroment
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
import numpy

import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import sys
cnn_num_argv = sys.argv
cnn_num = cnn_num_argv[1]

#******************************************************************************
#class num、class name、image_size
num_classes = 2
folder = ["0","1"]
model_number = 4

#epoch, batch_size
epochs = 20
batch_size = 4

#learning_num, data_set
learnig_num = 1
train_set_num = cnn_num
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

#path to training dataset
data_path = "./dataset/pre_train/" + "train_" + str(train_set_num)
training_path = data_path + "/training/"
validation_path = data_path + "/validation/"
testing_path = data_path + "/testing/"

"""## 2. Load dataset"""

x_train = []
y_train = []
x_test = []
y_test = []

for index, name in enumerate(folder):
    dir = training_path + name

    files = glob.glob(dir + "/*.png")
    for i, file in enumerate(files):
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
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        x_test.append(data)
        y_test.append(index)
        print("validation_" + str(name) + ":　"+ str(i))

x_test = np.array(x_test)
y_test = np.array(y_test)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

"""## 3. Training phase

### Building model
"""

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

if not os.path.exists("./model/cnn" + str(cnn_num)):
  os.makedirs("./model/cnn" + str(cnn_num))

# ModelCheckpoint
checkpoint = ModelCheckpoint(
                    filepath="./model/cnn" + str(cnn_num) + "/learning_" + str(learnig_num) + "_train_set_" + str(train_set_num) +  "_batch_" + str(batch_size) + "_{epoch:02d}.h5",
                    monitor='val_loss',
                    save_best_only=False,
                    period=1,
                )

# reduce learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, verbose=1,)


model.compile(loss='binary_crossentropy',
              optimizer=Adam(0.0001),
              metrics=['accuracy'])


model_name = "./model/best_model_cnn_" + str(cnn_num) + ".h5"
model.load_weights(model_name)

if not os.path.exists("./result/cnn" + str(cnn_num)):
  os.makedirs("./result/cnn" + str(cnn_num))
#-----------------------------------------------------------------------
def print_cmx(y_true, y_pred):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    plt.figure(figsize = (10,7))
    sns.heatmap(df_cmx, cmap="Blues", annot=True)
    plt.xlabel("Predicted_class")
    plt.ylabel("Correct_class")
    plt.savefig("./result/cnn" + str(cnn_num) + "/result_cnn" + str(cnn_num) + "_CM.tiff", format="tiff", dpi=300)
    plt.show()

def print_cmx_rate(y_true, y_pred):
    labels = [0, 1]
    cmx_data_1 = confusion_matrix(y_true, y_pred, labels=labels)

    cmx_data = [[0, 0],
                [0, 0]]

    for j in range(2):
      for i in range(2):
        percent = cmx_data_1[j][i] / (cmx_data_1[j][0] + cmx_data_1[j][1])
        cmx_data[j][i] = percent

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    plt.figure(figsize = (10,10))
    ax = sns.heatmap(df_cmx, cmap="Blues", annot=True, cbar = False, square = True)
    #ax.figure.axes[-1].yaxis.label.set_size(20)
    plt.rcParams["font.size"] = 40
    #sns.set_context("paper")

    plt.title("CNN" + str(cnn_num) + " model", fontsize=30)

    plt.xlabel("Predicted class", fontsize=35)
    plt.ylabel("Correct class", fontsize=35)
    #-------------------------------------------------------------------------
    plt.savefig("./result/cnn" + str(cnn_num) + "/result_cnn" + str(cnn_num) + "_CM_rate.tiff", format="tiff", dpi=300)
    #-------------------------------------------------------------------------
    plt.show()

predicted_class = []
prediction_y = model.predict(x_test)
for i in range(len(prediction_y)):
    if prediction_y[i][0] >= prediction_y[i][1]:
        predicted_class.append(0)
    elif prediction_y[i][1] >= prediction_y[i][0]:
        predicted_class.append(1)
    else:
        print("error")
print(predicted_class)


s_num = glob.glob(validation_path + '0/*.png')
l_num = glob.glob(validation_path + '1/*.png')
label_class = []
for i in range(len(s_num)):
    label_class.append(0)
for i in range(len(l_num)):
    label_class.append(1)

print(label_class)

print_cmx(label_class, predicted_class)
print_cmx(label_class, predicted_class)
print_cmx_rate(label_class, predicted_class)
print_cmx_rate(label_class, predicted_class)

result_matrix = np.empty((len(label_class)+1,3),dtype='U100')
result_matrix[0,0] = "correct_label"
result_matrix[0,1] = 'label_0'
result_matrix[0,2] = "label_1"

for i in range(len(label_class)):
    result_matrix[i+1,0] = str(label_class[i])
    result_matrix[i+1,1] = str(prediction_y[i][0])
    result_matrix[i+1,2] = str(prediction_y[i][1])

#----------------------------------------------------------------------------
np.savetxt("./result/cnn" + str(cnn_num) + "/result_cnn" + str(cnn_num) + ".csv" ,result_matrix, delimiter=',',fmt='%s')
#---------------------------------------------------------------------------
