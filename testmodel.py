
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from sklearn.utils import shuffle
from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
import cv2
import glob


size = 100

# LeNet-5 Model
def LeNet5():
    model = Sequential()
    
    # C1: Convolutional Layer
    model.add(Conv2D(6, (5, 5), activation='relu', input_shape=(size, size, 1), padding='valid'))
    
    # S2: Max-Pooling Layer
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    # C3: Convolutional Layer
    model.add(Conv2D(16, (5, 5), activation='relu', padding='valid'))
    
    # S4: Max-Pooling Layer
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    # Flatten the output
    model.add(Flatten())
    
    # C5: Fully Connected Layer
    model.add(Dense(120, activation='relu'))
    
    # F6: Fully Connected Layer
    model.add(Dense(84, activation='relu'))
    
    # Output Layer
    model.add(Dense(2, activation='softmax'))
    
    return model


# Load Training data: pothole
potholeTrainImages = glob.glob("./Pothole_detect/author/My Dataset/Pothole/*.jpg")


# potholeTrainImages = glob.glob("./Pothole_detect/author/My Dataset/train/Pothole/*.jpg")
# potholeTrainImages.extend(glob.glob("./Pothole_detect/author/My Dataset/train/Pothole/*.jpeg"))
# potholeTrainImages.extend(glob.glob("./Pothole_detect/author/My Dataset/train/Pothole/*.png"))

train1 = [cv2.imread(img, 0) for img in potholeTrainImages]
for i in range(0, len(train1)):
    train1[i] = cv2.resize(train1[i], (size, size))
temp1 = np.asarray(train1)

# Load Training data: non-pothole
nonPotholeTrainImages = glob.glob("./Pothole_detect/author/My Dataset/Plain/*.jpg")
train2 = [cv2.imread(img, 0) for img in nonPotholeTrainImages]
for i in range(0, len(train2)):
    train2[i] = cv2.resize(train2[i], (size, size))
temp2 = np.asarray(train2)

# Load Testing data: non-pothole
nonPotholeTestImages = glob.glob("./Pothole_detect/author/My Dataset/test/Plain/*.jpg")
test2 = [cv2.imread(img, 0) for img in nonPotholeTestImages]
for i in range(0, len(test2)):
    test2[i] = cv2.resize(test2[i], (size, size))
temp4 = np.asarray(test2)

# Load Testing data: potholes
potholeTestImages = glob.glob("./Pothole_detect/author/My Dataset/test/Pothole/*.jpg")
test1 = [cv2.imread(img, 0) for img in potholeTestImages]
for i in range(0, len(test1)):
    test1[i] = cv2.resize(test1[i], (size, size))
temp3 = np.asarray(test1)

X_train = np.concatenate([temp1, temp2])
X_test = np.concatenate([temp3, temp4])

y_train1 = np.ones([temp1.shape[0]], dtype=int)
y_train2 = np.zeros([temp2.shape[0]], dtype=int)
y_test1 = np.ones([temp3.shape[0]], dtype=int)
y_test2 = np.zeros([temp4.shape[0]], dtype=int)

y_train = np.concatenate([y_train1, y_train2])
y_test = np.concatenate([y_test1, y_test2])

X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)

X_train = X_train.reshape(X_train.shape[0], size, size, 1)
X_test = X_test.reshape(X_test.shape[0], size, size, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("Train data shape X:", X_train.shape)
print("Train data shape y:", y_train.shape)

inputShape = (size, size, 1)
model = LeNet5()

model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=15, validation_split=0.1)

metrics = model.evaluate(X_test, y_test)
for metric_name, metric_value in zip(model.metrics_names, metrics):
    print('{}: {}'.format(metric_name, metric_value))

print("Saving model weights and configuration file")

model.save('sampleLeNet-5.h5')

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
