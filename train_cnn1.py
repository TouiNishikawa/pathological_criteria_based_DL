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
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Flatten
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import efficientnet.keras as enet
import glob
import numpy as np
import os, cv2
from PIL import Image

#******************************************************************************
#class num、class name、image_size
num_classes = 2
folder = ["Low","High"]
model_number = 4

#epoch, batch_size
epochs = 15
batch_size = 4

#learning_num, data_set
learnig_num = 1
data_num = 6
train_set_num = 1
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
    dir = testing_path + name
    files = glob.glob(dir + "/*.png")
    for i, file in enumerate(files):
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

# ModelCheckpoint
checkpoint = ModelCheckpoint(
                    filepath="./model/cnn1/learning_" + str(learnig_num) + "_train_set_" + str(train_set_num) +  "_batch_" + str(batch_size) + "_{epoch:02d}.h5",
                    monitor='val_loss',
                    save_best_only=False,
                    period=1,
                )

# reduce learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, verbose=1,)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.0001),
              metrics=['accuracy'])

print("Training....")
hist = model.fit(x_train, y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 validation_data=(x_test,y_test),
                 callbacks=[checkpoint,reduce_lr],
                 shuffle=True,
                 verbose=1)
