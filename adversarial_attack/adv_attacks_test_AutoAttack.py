#!/usr/bin/env python
# coding: utf-8

# # 测试AutoAttack对抗样本在不同模型上的对抗迁移性

# In[2]:


import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np
import sys
sys.path.append('..')
sys.path.append('../..')
from adversarial_train import adv_train_dataset
from model.create_model import dataset
from model.models import MLP,DNN,LSTM
from model.create_model import train, utils, my_test
from logger import setup_logging
from sklearn.metrics import classification_report


# In[3]:


LOG_CONFIG_PATH = os.path.join(os.path.abspath(".."), "logger", "logger_config.json")
DATA_DIR  = os.path.join(os.path.abspath(".."), "data")
LOG_DIR   = os.path.join(os.path.abspath(".."), "logs")
DATA_DIR_ADV_EXAMPLES  = os.path.join(os.path.abspath(".."),'adversarial_attacks','adversarial_examples')
print(DATA_DIR_ADV_EXAMPLES)
TRAIN_DATA_DIR = os.path.join(os.path.abspath(".."),'data','processed')
print(TRAIN_DATA_DIR)
test_malicious_labels_path = f"{DATA_DIR}/processed/test/test_malicious_labels.pkl"


# 检查GPU

# In[4]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
utils.mkdir(LOG_DIR)
setup_logging(save_dir=LOG_DIR, log_config=LOG_CONFIG_PATH)


# ## 导入三个模型

# In[8]:


# 导入新的模型
model_MLP = MLP(49, 64, 64, 8)
model_MLP.to(device)
model_MLP.load_state_dict(torch.load('../model/create_model/created_models/MLP.pt'))

model_DNN = DNN()
model_DNN.to(device)
model_DNN.load_state_dict(torch.load('../model/create_model/created_models/DNN.pt'))

model_LSTM = LSTM(49,64,8,3)
model_LSTM.to(device)
model_LSTM.load_state_dict(torch.load('../model/create_model/created_models/LSTM.pt'))


# In[9]:


# 损失函数
criterion = nn.CrossEntropyLoss()
batch_size = 64
labels = ['Benign', 'Botnet ARES', 'Brute Force','DDoS','DoS', 'Infiltration','PortScan','Web Attack']


# ## 单独五种攻击需要分别生成对抗样本，测试攻击效果（代码略）

# ## 分别测试有目标和无目标的自定义攻击集合

# ### 测试无目标攻击 ['apgd-ce', 'apgd-dlr', 'fab']

# no-target + MLP

# In[10]:


adv_ex_features_auto_no_target_MLP = adv_train_dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/adv_ex_features_auto_no_target_MLP.pkl",   # 原攻击代码生成的对抗样本
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                                #转换成张量
            target_transform=torch.tensor)

# 加载对抗样本
test_loader = torch.utils.data.DataLoader(dataset=adv_ex_features_auto_no_target_MLP, batch_size=batch_size, shuffle=False)

# MLP模型分类准确率
history = my_test.test(model_MLP, criterion, test_loader, device)
# DNN模型分类准确率
history = my_test.test(model_DNN, criterion, test_loader, device)
# LSTM模型分类准确率
history = my_test.test(model_LSTM, criterion, test_loader, device)


# no-target + DNN

# In[13]:


adv_ex_features_auto_no_target_DNN = adv_train_dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/adv_ex_features_auto_no_target_DNN.pkl",   
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                                
            target_transform=torch.tensor)

# 加载对抗样本
test_loader = torch.utils.data.DataLoader(dataset=adv_ex_features_auto_no_target_DNN, batch_size=batch_size, shuffle=False)

history = my_test.test(model_MLP, criterion, test_loader, device)
history = my_test.test(model_DNN, criterion, test_loader, device)
history = my_test.test(model_LSTM, criterion, test_loader, device)


# no-target + LSTM

# In[ ]:


adv_ex_features_auto_no_target_LSTM = adv_train_dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/adv_ex_features_auto_no_target_LSTM.pkl",   
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                                
            target_transform=torch.tensor)

# 加载对抗样本
test_loader = torch.utils.data.DataLoader(dataset=adv_ex_features_auto_no_target_LSTM, batch_size=batch_size, shuffle=False)

history = my_test.test(model_MLP, criterion, test_loader, device)
history = my_test.test(model_DNN, criterion, test_loader, device)
history = my_test.test(model_LSTM, criterion, test_loader, device)


# ### 测试有目标攻击['apgd-t', 'fab-t']

# target + MLP

# In[19]:


adv_ex_features_auto_target_MLP = adv_train_dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/adv_ex_features_auto_target_MLP.pkl",  
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                                
            target_transform=torch.tensor)

# 加载对抗样本
test_loader = torch.utils.data.DataLoader(dataset=adv_ex_features_auto_target_MLP, batch_size=batch_size, shuffle=False)

history = my_test.test(model_MLP, criterion, test_loader, device)
history = my_test.test(model_DNN, criterion, test_loader, device)
history = my_test.test(model_LSTM, criterion, test_loader, device)


# target + DNN

# In[22]:


adv_ex_features_auto_target_DNN = adv_train_dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/adv_ex_features_auto_target_DNN.pkl",  
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                                
            target_transform=torch.tensor)

# 加载对抗样本
test_loader = torch.utils.data.DataLoader(dataset=adv_ex_features_auto_target_DNN, batch_size=batch_size, shuffle=False)

history = my_test.test(model_MLP, criterion, test_loader, device)
history = my_test.test(model_DNN, criterion, test_loader, device)
history = my_test.test(model_LSTM, criterion, test_loader, device)


# target + LSTM

# In[25]:


adv_ex_features_auto_target_LSTM = adv_train_dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/adv_ex_features_auto_target_LSTM.pkl",   
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                               
            target_transform=torch.tensor)

# 加载对抗样本
test_loader = torch.utils.data.DataLoader(dataset=adv_ex_features_auto_target_LSTM, batch_size=batch_size, shuffle=False)

history = my_test.test(model_MLP, criterion, test_loader, device)
history = my_test.test(model_DNN, criterion, test_loader, device)
history = my_test.test(model_LSTM, criterion, test_loader, device)

