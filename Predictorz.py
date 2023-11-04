# 导入必要的库
import numpy as np  # 用于处理数组和矩阵
import cv2  # OpenCV库，用于图像处理
import glob  # 用于查找文件路径
from keras.models import Sequential  # Keras深度学习库的Sequential模型
from keras.models import load_model  # 用于加载已训练的模型
from sklearn.preprocessing import LabelBinarizer  # 用于标签独热编码
from sklearn.model_selection import train_test_split  # 用于数据集划分
from keras.utils import np_utils  # 用于处理Keras模型的标签编码

# 定义图像大小
global size
size = 100  # 定义图像的大小为100x100像素

# 创建一个Keras模型
model = Sequential()

# 从磁盘加载已经训练好的模型
model = load_model('D:\\12in\\sample1.h5')

# 定义一个空的测试数据集
X_test = []

## 加载测试数据集：非坑洼道路的图像
nonPotholeTestImages = glob.glob("./Pothole_detect/author/My Dataset/test/Plain/*.jpg")

# 读取非坑洼道路的测试图像
test2 = [cv2.imread(img, 0) for img in nonPotholeTestImages]

# 调整图像的大小为指定的大小
for i in range(0, len(test2)):
    test2[i] = cv2.resize(test2[i], (size, size))

temp4 = np.asarray(test2)

## 加载测试数据集：坑洼道路的图像
potholeTestImages = glob.glob("./Pothole_detect/author/My Dataset/test/Pothole/*.jpg")

# 读取坑洼道路的测试图像
test1 = [cv2.imread(img, 0) for img in potholeTestImages]

# 调整图像的大小为指定的大小
for i in range(0, len(test1)):
    test1[i] = cv2.resize(test1[i], (size, size))

temp3 = np.asarray(test1)

# 将两个测试数据集合并成一个
X_test.extend(temp3)
X_test.extend(temp4)
X_test = np.asarray(X_test)

# 将测试数据的形状调整为适合模型输入的形状
X_test = X_test.reshape(X_test.shape[0], size, size, 1)

# 创建测试标签
y_test1 = np.ones([temp3.shape[0]], dtype=int)
y_test2 = np.zeros([temp4.shape[0]], dtype=int)

# 创建一个空列表y_test，用于存储所有的测试标签
y_test = []

# 将坑洼道路的测试标签和非坑洼道路的测试标签添加到y_test列表中
y_test.extend(y_test1)
y_test.extend(y_test2)

# 将y_test列表转换为NumPy数组，以便进行后续处理
y_test = np.asarray(y_test)

# 将测试标签进行独热编码
y_test = np_utils.to_categorical(y_test)

# 使用模型对测试数据进行分类预测
tests = model.predict_classes(X_test)

# 输出每个测试图像的分类预测结果
for i in range(len(X_test)):
    print(">>> 预测结果=%s" % (tests[i]))
