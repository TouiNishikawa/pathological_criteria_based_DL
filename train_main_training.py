# -*- coding: utf-8 -*-
"""train_main_training.ipynb
"""

"""## Load pre_trained models"""
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

"""### load cnn1"""

num_classes = 2
image_size = 380
model_name = "./model/best_model_cnn_1.h5"

from keras.backend import sigmoid
class SwishActivation(Activation):

    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'

def swish_act(x, beta = 1):
    return (x * sigmoid(beta * x))
from keras.utils.generic_utils import get_custom_objects
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
model.summary()

model.load_weights(model_name)
model.compile(
    optimizer = Adam(),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

model_1_1 = Model(model.get_layer(index=0).input, model.get_layer(index=-2).output)

"""### load cnn2"""

num_classes = 2
image_size = 380
model_name = "./model/best_model_cnn_2.h5"

from keras.backend import sigmoid
class SwishActivation(Activation):

    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'

def swish_act(x, beta = 1):
    return (x * sigmoid(beta * x))
from keras.utils.generic_utils import get_custom_objects
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
model.summary()

model.load_weights(model_name)
model.compile(
    optimizer = Adam(),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

model_2_1 = Model(model.get_layer(index=0).input, model.get_layer(index=-2).output)

"""### load cnn3"""

num_classes = 2
image_size = 380
model_name = "./model/best_model_cnn_3.h5"

from keras.backend import sigmoid
class SwishActivation(Activation):

    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'

def swish_act(x, beta = 1):
    return (x * sigmoid(beta * x))
from keras.utils.generic_utils import get_custom_objects
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
model.summary()

model.load_weights(model_name)
model.compile(
    optimizer = Adam(),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

model_3_1 = Model(model.get_layer(index=0).input, model.get_layer(index=-2).output)

"""### load cnn4"""

num_classes = 2
image_size = 380
model_name = "./model/best_model_cnn_4.h5"

from keras.backend import sigmoid
class SwishActivation(Activation):

    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'

def swish_act(x, beta = 1):
    return (x * sigmoid(beta * x))
from keras.utils.generic_utils import get_custom_objects
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
model.summary()

model.load_weights(model_name)
model.compile(
    optimizer = Adam(),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

model_4_1 = Model(model.get_layer(index=0).input, model.get_layer(index=-2).output)

"""### load cnn5"""

num_classes = 2
image_size = 380
model_name = "./model/best_model_cnn_5.h5"

from keras.backend import sigmoid
class SwishActivation(Activation):

    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'

def swish_act(x, beta = 1):
    return (x * sigmoid(beta * x))
from keras.utils.generic_utils import get_custom_objects
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
model.summary()

model.load_weights(model_name)
model.compile(
    optimizer = Adam(),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

model_5_1 = Model(model.get_layer(index=0).input, model.get_layer(index=-2).output)

"""### load cnn6"""

num_classes = 2
image_size = 380
model_name = "./model/best_model_cnn_6.h5"

from keras.backend import sigmoid
class SwishActivation(Activation):

    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'

def swish_act(x, beta = 1):
    return (x * sigmoid(beta * x))
from keras.utils.generic_utils import get_custom_objects
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
model.summary()

model.load_weights(model_name)
model.compile(
    optimizer = Adam(),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

model_6_1 = Model(model.get_layer(index=0).input, model.get_layer(index=-2).output)

"""## main training"""

#******************************************************************************
num_classes = 4
batch_size = 4
epochs = 15
folder = ["Normal","Atypical","Dysplasia","CIS"]
data_path ="./dataset/main_train"
training_path = data_path + "/training/"
validation_path = data_path+ "/validation/"
testing_path = data_path + "/testing/"
#******************************************************************************

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
        oks = glob.glob(data_path + '/validation/Normal/*.png')
        ngs = glob.glob(data_path + '/validation/Atypical/*.png') + glob.glob(data_path + '/validation/Dysplasia/*.png') + glob.glob(data_path + '/validation/CIS/*.png')
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

def read_and_preprocess_img(path, size=(image_size,image_size)):
    img = load_img(path, target_size=size)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x

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

"""## load training and testing dataset"""

#load data set
DATA = data()
x_train_normal, x_ref, y_ref, x_test_normal, x_test_anomaly, normal_path_test, anomaly_path_test = DATA.get_data()

#train_data_set
normal_label = np.zeros((len(x_train_normal), 4))
normal_label[:,0] = 1
anomaly_label = np.zeros((len(x_ref), 4))

x = np.vstack((x_train_normal, x_ref))
y = np.vstack((normal_label, y_ref))

#validation_data_set
normal_label = np.zeros((len(x_test_normal), 4))
normal_label[:,0] = 1
anomaly_label = np.zeros((len(x_test_anomaly), 4))

ng_folder = ["Atypical","Dysplasia","CIS"]
count_all = 0
end = 0
for index, name in enumerate(ng_folder):
  dir = validation_path + name
  files = glob.glob(dir + "/*.png")
  for i, file in enumerate(files):
    anomaly_label[end + i][index+1] = 1
    count_all += 1
  end = count_all
x_val = np.vstack((x_test_normal,x_test_anomaly))
y_val = np.vstack((normal_label, anomaly_label))


classes =  y_ref.shape[1]

"""### training"""

print("training...")
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001, amsgrad=True),
              metrics=['accuracy'])

# checkpoint
checkpoint = ModelCheckpoint(
                    filepath="main_trained_model"  +  "_batch_" + str(batch_size) + "_{epoch:02d}.h5",
                    monitor='val_loss',
                    save_best_only=False,
                    period=1,
                )

es_cb = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=10, verbose=1, mode='auto')

hist = model.fit([x, x, x, x, x, x], y, batch_size=batch_size, epochs=epochs, verbose = 1, callbacks=[es_cb, checkpoint], validation_data=([x_val,x_val,x_val,x_val,x_val,x_val], y_val))
model.save_weights(model_naming)
