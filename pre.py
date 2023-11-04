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
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, auc, precision_recall_curve, confusion_matrix, log_loss
import matplotlib.pyplot as plt
from matplotlib import rc

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle

import cv2
import glob

global size
size = 100
model = Sequential()
rc("font",family='YouYuan')
#sample.h5路径
#model = load_model('.\\Pothole_detect\\author\\sample.h5')
#model = load_model('.\\sample.h5')
model = load_model('.\\sampleLeNet-5.h5')
#test图片路径
nonPotholeTestImages = glob.glob("./Pothole_detect/author/My Dataset/TEST_competiton/*.jpg")
#test图片路径
directory_path = "./Pothole_detect/author/My Dataset/TEST_competiton/"
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

true_labels = [0 if 'normal' in fname else 1 for fname in file_names]
predicted_labels = tests

# 计算准确性
accuracy = accuracy_score(true_labels, predicted_labels)

# 计算精确度
precision = precision_score(true_labels, predicted_labels)

# 计算召回率
recall = recall_score(true_labels, predicted_labels)

# 计算 F1 分数
f1 = f1_score(true_labels, predicted_labels)

# 计算ROC曲线和AUC
fpr, tpr, thresholds = roc_curve(true_labels, predicted_labels)
roc_auc = auc(fpr, tpr)

# 计算查准率-召回率曲线和AUC
precision_, recall_, _ = precision_recall_curve(true_labels, predicted_labels)
pr_auc = auc(recall_, precision_)

# 计算混淆矩阵
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# 计算对数损失
logloss = log_loss(true_labels, predicted_labels)

# 打印性能评估结果
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC曲线下的面积 (AUC)：{roc_auc}")
print(f"查准率-召回率曲线下的面积 (AUC)：{pr_auc}")
print("混淆矩阵：")
print(conf_matrix)
print(f"对数损失：{logloss}")
#print(true_labels)

#df = pd.DataFrame(data)
# 将 DataFrame 保存到 Excel 文件中
#df.to_excel('./test_resulttest.xlsx', index=False)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC 曲线 (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正例率 (FPR)')
plt.ylabel('真正例率 (TPR)')
plt.title('ROC 曲线')
plt.legend(loc="lower right")
plt.show()


# 绘制查准率-召回率曲线
plt.figure()
plt.step(precision_, recall_, color='b', where='post')
plt.fill_between(precision_, recall_, step='post', alpha=0.2, color='b')
plt.xlabel('召回率 (Recall)')
plt.ylabel('查准率 (Precision)')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('查准率-召回率曲线')
plt.show()

def plot_confusion_matrix(confusion_mat):  
    '''''将混淆矩阵画图并显示出来'''  
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.gray)  
    plt.title('Confusion matrix')  
    plt.colorbar()  
    tick_marks = np.arange(confusion_mat.shape[0])  
    plt.xticks(tick_marks, tick_marks)  
    plt.yticks(tick_marks, tick_marks)  
    plt.ylabel('True label')  
    plt.xlabel('Predicted label')  
    plt.show()  
  
plot_confusion_matrix(conf_matrix)  