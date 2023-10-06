#!/usr/bin/env python
# coding: utf-8

# # 搭建MLP模型（使用平衡的训练数据集）

# In[ ]:

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append("..")
sys.path.append("../..")

from logger import logger
from models import MLP
import dataset,train,utils,visualisation,my_test

from sklearn.metrics import classification_report


# In[ ]:

LOG_CONFIG_PATH = os.path.join(os.path.abspath("../.."), "logger", "logger_config.json")
LOG_DIR   = os.path.join(os.path.abspath("../.."), "logs")   # 存储日志信息路径
DATA_DIR  = os.path.join(os.path.abspath("../.."), "data")   # 数据集路径
IMAGE_DIR = os.path.join(os.path.abspath("../.."), "images")  # 存储的图像结果路径


# ### 检查GPU是否可用

# In[ ]:

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print('Using {} device'.format(device))

# 建立日志目录，设置日志配置
utils.mkdir(LOG_DIR)
logger.setup_logging(save_dir=LOG_DIR, log_config=LOG_CONFIG_PATH)


# ### 建立数据集加载器

# In[ ]:

# 加载数据集
train_data, val_data, test_data = dataset.get_dataset(data_path=DATA_DIR, balanced=True)

print('训练集中的样本数: ', len(train_data))
print('验证集中的样本数: ', len(val_data))
print('测试集中的样本数: ', len(test_data))
print('保留的特征数:',len(train_data.features.columns)) 
print('标签种类数:',len(train_data.labels.value_counts())) 

batch_size = 64

#创建数据加载器，加载训练、验证和测试集
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


# ### 实例化网络，设置损失函数和优化器

# In[ ]:

# 建立MLP
model = MLP(49, 64, 64, 8)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 优化器
learning_rate = 0.02
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Epochs 轮数
num_epochs = 10


# ### 训练模型

# In[ ]:

history = train.train(model, criterion, optimizer, train_loader, valid_loader, num_epochs, device)

training_loss = history['train']['loss']
training_accuracy = history['train']['accuracy']
train_output_true = history['train']['output_true']
train_output_pred = history['train']['output_pred']

validation_loss = history['valid']['loss']
validation_accuracy = history['valid']['accuracy']
valid_output_true = history['valid']['output_true']
valid_output_pred = history['valid']['output_pred']


# ### 绘制损失和准确率

# In[ ]:

fig = plt.figure(figsize=(12, 8))
plt.plot(training_loss, label='train - loss')
plt.plot(validation_loss, label='validation - loss')
plt.title("Train and Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="best")  #自动选择最好的图例位置
plt.show()

fig = plt.figure(figsize=(12, 8))
plt.plot(training_accuracy, label='train - accuracy')
plt.plot(validation_accuracy, label='validation - accuracy')
plt.title("Train and Validation Accuracy")
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.legend(loc="best")
plt.show()


# ### 绘制混淆矩阵 confusion matrix

# In[ ]:

labels = ['Benign', 'Botnet ARES', 'Brute Force','DDoS','DoS', 'Infiltration','PortScan','Web Attack']

visualisation.plot_confusion_matrix(y_true=train_output_true,
                                    y_pred=train_output_pred,
                                    labels=labels,
                                    save=True,
                                    save_dir=IMAGE_DIR,
                                    filename="mlp_train_confusion_matrix.pdf")


# In[ ]:

print("Training Set -- Classification Report", end="\n\n")
print(classification_report(train_output_true, train_output_pred, target_names=labels))


# In[ ]:

visualisation.plot_confusion_matrix(y_true=valid_output_true,
                                    y_pred=valid_output_pred,
                                    labels=labels,
                                    save=True,
                                    save_dir=IMAGE_DIR,
                                    filename="mlp_valid_confusion_matrix.pdf")


# In[ ]:

print("Validation Set -- Classification Report", end="\n\n")
print(classification_report(valid_output_true, valid_output_pred, target_names=labels))


# ### 保存模型

# In[ ]:
path = 'created_models/MLP.pt'
torch.save( model.state_dict(), path)


# ### 加载已保存模型，利用测试集测试

# In[ ]:

model = MLP(49, 64, 64, 8)
model.to(device)
model.load_state_dict(torch.load('created_models/MLP.pt'))
history = my_test.test(model, criterion, test_loader, device)

test_output_true = history['test']['output_true']
test_output_pred = history['test']['output_pred']
test_output_pred_prob = history['test']['output_pred_prob']


# ### 分类报告

# In[ ]:

labels = ['Benign', 'Botnet ARES', 'Brute Force','DDoS','DoS', 'Infiltration','PortScan','Web Attack']
visualisation.plot_confusion_matrix(y_true=test_output_true,
                                    y_pred=test_output_pred,
                                    labels=labels,
                                    save=True,
                                    save_dir=IMAGE_DIR,
                                    filename="mlp_test_confusion_matrix.svg")


# In[ ]:

print("Testing Set -- Classification Report", end="\n\n")
print(classification_report(test_output_true, test_output_pred, target_names=labels))


# ### 绘制 ROC 曲线

# In[ ]:

y_test = pd.get_dummies(test_output_true).values
y_score = np.array(test_output_pred_prob)

visualisation.plot_roc_curve(y_test=y_test,
                             y_score=y_score,
                             labels=labels,
                             save=True,
                             save_dir=IMAGE_DIR,
                             filename="mlp_roc_curve.pdf")


# ### 绘制 精确率 (precision）与 召回率  (recall)  曲线

# In[ ]:

visualisation.plot_precision_recall_curve(y_test=y_test,
                                          y_score=y_score,
                                          labels=labels,
                                          save=True,
                                          save_dir=IMAGE_DIR,
                                          filename="mlp_prec_recall_curve.pdf")

