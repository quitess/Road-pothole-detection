import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import tensorflow as tf
from tensorflow.contrib.layers import flatten

from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, Callback
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, ELU,GlobalAveragePooling2D, regularizers
from keras.layers.convolutional import Convolution2D, Cropping2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import adam
from sklearn.utils import shuffle
from keras.utils import np_utils

from tensorflow.keras.layers import BatchNormalization


import cv2
import glob
size = 100

# AlexNet Model
def kerasModel4():
    model = Sequential()

    # Layer 1
    model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=(size, size, 1), padding='valid', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # Layer 2
    model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # Layer 3
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))

    # Layer 4
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))

    # Layer 5
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # Flatten the output
    model.add(Flatten())

    # Layer 6
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    # Layer 7
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(2, activation='softmax'))

    return model


size=100

 ## load Training data : pothole
potholeTrainImages = glob.glob("./Pothole_detect/author/My Dataset/train/Pothole/*.jpg")
potholeTrainImages.extend(glob.glob("./Pothole_detect/author/My Dataset/train/Pothole/*.jpeg"))
potholeTrainImages.extend(glob.glob("./Pothole_detect/author/My Dataset/train/Pothole/*.png"))



train1 = [cv2.imread(img,0) for img in potholeTrainImages]
for i in range(0,len(train1)):
    train1[i] = cv2.resize(train1[i],(size,size))
temp1 = np.asarray(train1)


#  ## load Training data : non-pothole
nonPotholeTrainImages = glob.glob("./Pothole_detect/author/My Dataset/train/Plain/*.jpg")

train2 = [cv2.imread(img,0) for img in nonPotholeTrainImages]
# train2[train2 != np.array(None)]
for i in range(0,len(train2)):
    train2[i] = cv2.resize(train2[i],(size,size))
temp2 = np.asarray(train2)



## load Testing data : non-pothole
nonPotholeTestImages = glob.glob("./Pothole_detect/author/My Dataset/test/Plain/*.jpg")

test2 = [cv2.imread(img,0) for img in nonPotholeTestImages]
# train2[train2 != np.array(None)]
for i in range(0,len(test2)):
    test2[i] = cv2.resize(test2[i],(size,size))
temp4 = np.asarray(test2)


## load Testing data : potholes
potholeTestImages = glob.glob("./Pothole_detect/author/My Dataset/test/Pothole/*.jpg")

test1 = [cv2.imread(img,0) for img in potholeTestImages]
# train2[train2 != np.array(None)]
for i in range(0,len(test1)):
    test1[i] = cv2.resize(test1[i],(size,size))
temp3 = np.asarray(test1)


X_train = []
X_train.extend(temp1)
X_train.extend(temp2)
X_train = np.asarray(X_train)

X_test = []
X_test.extend(temp3)
X_test.extend(temp4)
X_test = np.asarray(X_test)


y_train1 = np.ones([temp1.shape[0]],dtype = int)
y_train2 = np.zeros([temp2.shape[0]],dtype = int)
y_test1 = np.ones([temp3.shape[0]],dtype = int)
y_test2 = np.zeros([temp4.shape[0]],dtype = int)

print(y_train1[0])
print(y_train2[0])
print(y_test1[0])
print(y_test2[0])

y_train = []
y_train.extend(y_train1)
y_train.extend(y_train2)
y_train = np.asarray(y_train)

y_test = []
y_test.extend(y_test1)
y_test.extend(y_test2)
y_test = np.asarray(y_test)


X_train,y_train = shuffle(X_train,y_train)
X_test,y_test = shuffle(X_test,y_test)

# X_train.reshape([-1,50,50,1])
# X_test.reshape([-1,50,50,1])/
X_train = X_train.reshape(X_train.shape[0], size, size, 1)
X_test = X_test.reshape(X_test.shape[0], size, size, 1)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


print("train shape X", X_train.shape)
print("train shape y", y_train.shape)

inputShape = (size, size, 1)
model = kerasModel4()


model.compile('adam', 'categorical_crossentropy', ['accuracy'])
# model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=100,validation_split=0.1)

metrics = model.evaluate(X_test, y_test)
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))

print("Saving model weights and configuration file")

model.save('sampleLenet.h5')

model_json = model.to_json()
with open("truesample.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("truesample.h5")
print("Saved model to disk")



print(history.history)
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']

plt.figure(figsize=(10,3))

plt.subplot(121)
plt.plot(loss,color='b',label='train')
plt.plot(val_loss,color='r',label='test')
plt.ylabel('loss')
plt.legend()
plt.subplot(122)
plt.plot(acc,color='b',label='train')
plt.plot(val_acc,color='r',label='test')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

