import numpy as np
import cv2
import glob
from keras.models import Sequential
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import os
import pandas as pd
import openpyxl
global size
size = 100
model = Sequential()

#sample.h5路径
#model = load_model('.\\Pothole_detect\\author\\sample.h5')


model = load_model('.\\sample2.0.h5')

#test图片路径
#nonPotholeTestImages = glob.glob("./Pothole_detect/author/My Dataset/TEST_competiton/*.jpg")

nonPotholeTestImages = glob.glob("./Pothole_detect/author/My Dataset/testdata_V2/*.jpg")

#test图片路径
#directory_path = "./Pothole_detect/author/My Dataset/TEST_competiton/"

directory_path = "./Pothole_detect/author/My Dataset/testdata_V2/"

file_names = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

test2 = [cv2.imread(img,0) for img in nonPotholeTestImages]
for i in range(0,len(test2)):
    test2[i] = cv2.resize(test2[i],(size,size))
temp4 = np.asarray(test2)



X_test = []
X_test.extend(temp4)
X_test = np.asarray(X_test)

X_test = X_test.reshape(X_test.shape[0], size, size, 1)

tests = model.predict_classes(X_test)

num_to_add = len(tests) - len(file_names)
    # 添加空字符串到 fnames 列表末尾
file_names.extend([''] * num_to_add)

data = {'fnames': file_names, 'label': tests}

# n = 0
# for i in data['fnames']:
#     print(f"filename: {i}, label: {data['label'][n]}")
#     n +=1

df = pd.DataFrame(data)
# 将 DataFrame 保存到 Excel 文件中
df.to_excel('./test_result.xlsx', index=False)