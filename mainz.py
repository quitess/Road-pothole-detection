import pandas as pd  # 导入Pandas库用于数据处理
import numpy as np  # 导入NumPy库用于数值计算
import matplotlib.pyplot as plt  # 导入Matplotlib库用于绘图
import matplotlib.mlab as mlab  # 导入Matplotlib的mlab模块

import tensorflow as tf  # 导入TensorFlow库
from tensorflow.contrib.layers import flatten  # 从TensorFlow的contrib模块中导入flatten函数

from keras.layers.pooling import MaxPooling2D  # 从Keras库中导入MaxPooling2D层
from keras.models import Sequential, Model  # 导入Keras的Sequential模型和Model类
from keras.callbacks import EarlyStopping, Callback  # 导入Keras的EarlyStopping回调和Callback类
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, ELU, GlobalAveragePooling2D, regularizers  # 导入Keras的不同层和激活函数
from keras.layers.convolutional import Convolution2D, Cropping2D, Conv2D  # 导入Keras的卷积层
from keras.layers.pooling import MaxPooling2D  # 导入Keras的MaxPooling2D层
from keras.optimizers import adam  # 导入Keras的Adam优化器
from sklearn.utils import shuffle  # 从sklearn库导入shuffle函数
from keras.utils import np_utils  # 从Keras库导入np_utils模块，用于独热编码

import time, cv2, glob  # 导入时间模块、OpenCV库和glob模块

global inputShape, size  # 声明全局变量inputShape和size

def kerasModel4():  # 定义一个Keras模型
    model = Sequential()  # 创建一个Sequential模型


    model.add(Conv2D(16, (8, 8), strides=(4, 4), padding='valid', input_shape=(size, size, 1))  # 添加一个卷积层，设置输入形状和卷积核参数
    model.add(Activation('relu'))  # 添加ReLU激活函数

    model.add(Conv2D(32, (5, 5), padding="same"))  # 添加另一个卷积层
    model.add(Activation('relu'))  # 添加ReLU激活函数

    model.add(GlobalAveragePooling2D())  # 添加全局平均池化层

    model.add(Dense(512))  # 添加全连接层
    model.add(Dropout(.1))  # 添加Dropout层
    model.add(Activation('relu'))  # 添加ReLU激活函数

    model.add(Dense(2))  # 添加输出层，输出维度为2
    model.add(Activation('softmax'))  # 使用Softmax激活函数

    return model

size = 100  # 设置图像大小为100x100像素

# 加载训练数据：坑洼
potholeTrainImages = glob.glob("./Pothole_detect/author/My Dataset/train/Pothole/*.jpg")  # 使用glob查找坑洼训练图像文件
potholeTrainImages.extend(glob.glob("./Pothole_detect/author/My Dataset/train/Pothole/*.jpeg"))  # 添加JPEG格式的坑洼训练图像
potholeTrainImages.extend(glob.glob("./Pothole_detect/author/My Dataset/train/Pothole/*.png"))  # 添加PNG格式的坑洼训练图像

train1 = [cv2.imread(img, 0) for img in potholeTrainImages]  # 使用OpenCV读取坑洼训练图像，并将其转换为灰度图像
for i in range(0, len(train1)):  # 遍历所有图像
    train1[i] = cv2.resize(train1[i], (size, size))  # 调整图像大小为指定的尺寸
temp1 = np.asarray(train1)  # 将图像数据转换为NumPy数组

# 加载训练数据：非坑洼
nonPotholeTrainImages = glob.glob("./Pothole_detect/author/My Dataset/train/Plain/*.jpg")  # 使用glob查找非坑洼训练图像文件
train2 = [cv2.imread(img, 0) for img in nonPotholeTrainImages]  # 使用OpenCV读取非坑洼训练图像，并将其转换为灰度图像
for i in range(0, len(train2)):  # 遍历所有图像
    train2[i] = cv2.resize(train2[i], (size, size))  # 调整图像大小为指定的尺寸
temp2 = np.asarray(train2)  # 将图像数据转换为NumPy数组

# 加载测试数据：非坑洼
nonPotholeTestImages = glob.glob("./Pothole_detect/author/My Dataset/test/Plain/*.jpg")  # 使用glob查找非坑洼测试图像文件
test2 = [cv2.imread(img, 0) for img in nonPotholeTestImages]  # 使用OpenCV读取非坑洼测试图像，并将其转换为灰度图像
for i in range(0, len(test2)):  # 遍历所有图像
    test2[i] = cv2.resize(test2[i], (size, size))  # 调整图像大小为指定的尺寸
temp4 = np.asarray(test2)  # 将图像数据转换为NumPy数组

# 加载测试数据：坑洼
potholeTestImages = glob.glob("./Pothole_detect/author/My Dataset/test/Pothole/*.jpg")  # 使用glob查找坑洼测试图像文件
test1 = [cv2.imread(img, 0) for img in potholeTestImages]  # 使用OpenCV读取坑洼测试图像，并将其转换为灰度图像
for i in range(0, len(test1)):  # 遍历所有图像
    test1[i] = cv2.resize(test1[i], (size, size))  # 调整图像大小为指定的尺寸
temp3 = np.asarray(test1)  # 将图像数据转换为NumPy数组

X_train = []  # 初始化训练数据列表
X_train.extend(temp1)  # 将坑洼训练数据添加到列表中
X_train.extend(temp2)  # 将非坑洼训练数据添加到列表中
X_train = np.asarray(X_train)  # 将训练数据转换为NumPy数组

X_test = []  # 初始化测试数据列表
X_test.extend(temp3)  # 将坑洼测试数据添加到列表中
X_test.extend(temp4)  # 将非坑洼测试数据添加到列表中
X_test = np.asarray(X_test)  # 将测试数据转换为NumPy数组

y_train1 = np.ones([temp1.shape[0]], dtype=int)  # 创建标签数组，坑洼类别为1
y_train2 = np.zeros([temp2.shape[0]], dtype=int)  # 创建标签数组，非坑洼类别为0
y_test1 = np.ones([temp3.shape[0]], dtype=int)  # 创建标签数组，坑洼类别为1
y_test2 = np.zeros([temp4.shape[0]], dtype=int)  # 创建标签数组，非坑洼类别为0

y_train = []  # 初始化训练标签列表
y_train.extend(y_train1)  # 将坑洼训练标签添加到列表中
y_train.extend(y_train2)  # 将非坑洼训练标签添加到列表中
y_train = np.asarray(y_train)  # 将训练标签转换为NumPy数组

y_test = []  # 初始化测试标签列表
y_test.extend(y_test1)  # 将坑洼测试标签添加到列表中
y_test.extend(y_test2)  # 将非坑洼测试标签添加到列表中
y_test = np.asarray(y_test)  # 将测试标签转换为NumPy数组

X_train, y_train = shuffle(X_train, y_train)  # 对训练数据和标签进行随机排序
X_test, y_test = shuffle(X_test, y_test)  # 对测试数据和标签进行随机排序

X_train = X_train.reshape(X_train.shape[0], size, size, 1)  # 调整训练数据的形状
X_test = X_test.reshape(X_test.shape[0], size, size, 1)  # 调整测试数据的形状

y_train = np_utils.to_categorical(y_train)  # 使用独热编码对训练标签进行编码
y_test = np_utils.to_categorical(y_test)  # 使用独热编码对测试标签进行编码

print("训练数据形状 X", X_train.shape)  # 打印训练数据的形状
print("训练数据形状 y", y_train.shape)  # 打印训练标签的形状

inputShape = (size, size, 1)  # 设置输入形状
model = kerasModel4()  # 创建Keras模型

model.compile('adam', 'categorical_crossentropy', ['accuracy'])  # 编译模型，使用Adam优化器和交叉熵损失函数
history = model.fit(X_train, y_train, epochs=500, validation_split=0.1)  # 训练模型并记录历史信息

metrics = model.evaluate(X_test, y_test)  # 对测试数据进行评估
for metric_i in range(len(model.metrics_names)):  # 遍历模型的不同指标
    metric_name = model.metrics_names[metric_i]  # 获取指标名称
    metric_value = metrics[metric_i]  # 获取指标值
    print('{}: {}'.format(metric_name, metric_value))  # 打印指标名称和值

print("保存模型权重和配置文件")

model.save('sample2.h5')  # 保存模型权重
model_json = model.to_json()  # 将模型转换为JSON格式
with open("truesample.json", "w") as json_file:  # 将模型的JSON表示写入文件
    json_file.write(model_json)

model.save_weights("truesample.h5")  # 保存模型权重
print("模型已保存到磁盘")
