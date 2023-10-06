#!/usr/bin/env python
# coding: utf-8

# # 测试SI_NI_FGSM对抗样本在不同模型上的对抗迁移性

# In[ ]:


import torch
import torch.nn as nn
import os
import sys
sys.path.append('..')
sys.path.append('../..')

from adversarial_train import adv_train_dataset
from model.models import MLP,DNN,LSTM
from model.create_model import utils,my_test
from logger import setup_logging


# In[ ]:


LOG_CONFIG_PATH = os.path.join(os.path.abspath(".."), "logger", "logger_config.json")
DATA_DIR  = os.path.join(os.path.abspath(".."), "data")
LOG_DIR   = os.path.join(os.path.abspath(".."), "logs")
DATA_DIR_ADV_EXAMPLES  = os.path.join(os.path.abspath(".."),'adversarial_attacks','adversarial_examples')
print(DATA_DIR_ADV_EXAMPLES)
TRAIN_DATA_DIR = os.path.join(os.path.abspath(".."),'data','processed')
print(TRAIN_DATA_DIR)


# In[ ]:


# 检查GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
utils.mkdir(LOG_DIR)
setup_logging(save_dir=LOG_DIR, log_config=LOG_CONFIG_PATH)


# In[ ]:


test_malicious_labels_path = f"{DATA_DIR}/processed/test/test_malicious_labels.pkl"


# ## 导入三个模型

# In[ ]:


model_MLP = MLP(49, 64, 64, 8)
model_MLP.to(device)
model_MLP.load_state_dict(torch.load('../model/create_model/created_models/MLP.pt'))

model_DNN = DNN()
model_DNN.to(device)
model_DNN.load_state_dict(torch.load('../model/create_model/created_models/DNN.pt'))

model_LSTM = LSTM(49,64,8,3)
model_LSTM.to(device)
model_LSTM.load_state_dict(torch.load('../model/create_model/created_models/LSTM.pt'))


# ## SI_NI_FGSM+MLP生成的对抗样本

# In[ ]:


adv_ex_data_SI_NI_FGSM_MLP = adv_train_dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/adv_ex_features_SI_NI_FGSM_MLP.pkl",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                                
            target_transform=torch.tensor)

# 损失函数
criterion = nn.CrossEntropyLoss()
batch_size = 64
# 加载对抗样本
test_loader = torch.utils.data.DataLoader(dataset=adv_ex_data_SI_NI_FGSM_MLP, batch_size=batch_size, shuffle=False)

# MLP模型分类准确率
history = my_test.test(model_MLP, criterion, test_loader, device)
# DNN模型分类准确率
history = my_test.test(model_DNN, criterion, test_loader, device)
# LSTM模型分类准确率
history = my_test.test(model_LSTM, criterion, test_loader, device)


# ## SI_NI_FGSM+DNN生成的对抗样本

# In[ ]:


adv_ex_data_SI_NI_FGSM_DNN = adv_train_dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/adv_ex_features_SI_NI_FGSM_DNN.pkl",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                                
            target_transform=torch.tensor)

# 加载对抗样本
test_loader = torch.utils.data.DataLoader(dataset=adv_ex_data_SI_NI_FGSM_DNN, batch_size=batch_size, shuffle=False)

# MLP模型分类准确率
history = my_test.test(model_MLP, criterion, test_loader, device)
# DNN模型分类准确率
history = my_test.test(model_DNN, criterion, test_loader, device)
# LSTM模型分类准确率
history = my_test.test(model_LSTM, criterion, test_loader, device)


# ## SI_NI_FGSM+LSTM生成的对抗样本

# In[ ]:


adv_ex_data_SI_NI_FGSM_LSTM = adv_train_dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/adv_ex_features_SI_NI_FGSM_LSTM.pkl",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                                
            target_transform=torch.tensor)

# 加载对抗样本
test_loader = torch.utils.data.DataLoader(dataset=adv_ex_data_SI_NI_FGSM_LSTM, batch_size=batch_size, shuffle=False)

# 使用对抗训练后的新模型，重新分类对抗样本（FGSM_MLP生成）
# MLP模型分类准确率
history = my_test.test(model_MLP, criterion, test_loader, device)
# DNN模型分类准确率
history = my_test.test(model_DNN, criterion, test_loader, device)
# LSTM模型分类准确率
history = my_test.test(model_LSTM, criterion, test_loader, device)

