#!/usr/bin/env python
# coding: utf-8

# # 将SI_NI_FGSM+MLP, SI_NI_FGSM+DNN, SI_NI_FGSM+LSTM新对抗样本一同添加至原始训练数据集，并训练MLP/DNN/LSTM对抗模型，测试效率

# In[ ]:


import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np
import sys
sys.path.append('..')

from model.create_model import dataset
from model.models import MLP,DNN,LSTM
from model.create_model import train, utils
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


utils.mkdir(LOG_DIR)
setup_logging(save_dir=LOG_DIR, log_config=LOG_CONFIG_PATH)

# 检测GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


# ## 加载对抗样本

# In[ ]:


DATA_DIR_ADV_EXAMPLES  = os.path.join(os.path.abspath(".."),'adversarial_attacks','adversarial_examples')
DATA_DIR  = os.path.join(os.path.abspath(".."), "data")
print(DATA_DIR_ADV_EXAMPLES)
test_malicious_labels_path = f"{DATA_DIR}/processed/test/test_malicious_labels.pkl"
under_train_labels_path = f"{DATA_DIR}/processed/train/under_sampler_train_labels_balanced.pkl"

# 损失函数
criterion = nn.CrossEntropyLoss()
batch_size = 64


# ### FGSM对抗样本

# In[ ]:


# FGSM+MLP生成的对抗样本
adv_ex_data_FGSM_MLP = dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/adv_ex_features_FGSM_MLP.pkl",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                                
            target_transform=torch.tensor)

test_loader_FGSM_MLP = torch.utils.data.DataLoader(dataset=adv_ex_data_FGSM_MLP, batch_size=batch_size, shuffle=False)

# FGSM+DNN生成的对抗样本
adv_ex_data_FGSM_DNN = dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/adv_ex_features_FGSM_DNN.pkl",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                              
            target_transform=torch.tensor)

test_loader_FGSM_DNN = torch.utils.data.DataLoader(dataset=adv_ex_data_FGSM_DNN, batch_size=batch_size, shuffle=False)

# FGSM+LSTM生成的对抗样本
adv_ex_data_FGSM_LSTM = dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/adv_ex_features_FGSM_LSTM.pkl",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                               
            target_transform=torch.tensor)

test_loader_FGSM_LSTM = torch.utils.data.DataLoader(dataset=adv_ex_data_FGSM_LSTM, batch_size=batch_size, shuffle=False)


# ### NI_FGSM 对抗样本

# In[ ]:


# NI_FGSM+MLP生成的对抗样本
adv_ex_data_NI_FGSM_MLP = dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/adv_ex_features_NI_FGSM_MLP.pkl",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                                
            target_transform=torch.tensor)

test_loader_NI_FGSM_MLP = torch.utils.data.DataLoader(dataset=adv_ex_data_NI_FGSM_MLP, batch_size=batch_size, shuffle=False)

# NI_FGSM+DNN生成的对抗样本
adv_ex_data_NI_FGSM_DNN = dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/adv_ex_features_NI_FGSM_DNN.pkl",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                               
            target_transform=torch.tensor)

test_loader_NI_FGSM_DNN = torch.utils.data.DataLoader(dataset=adv_ex_data_NI_FGSM_DNN, batch_size=batch_size, shuffle=False)

# NI_FGSM+LSTM生成的对抗样本
adv_ex_data_NI_FGSM_LSTM = dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/adv_ex_features_NI_FGSM_LSTM.pkl",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                               
            target_transform=torch.tensor)

test_loader_NI_FGSM_LSTM = torch.utils.data.DataLoader(dataset=adv_ex_data_NI_FGSM_LSTM, batch_size=batch_size, shuffle=False)


# ### SI_NI_FGSM 对抗样本

# In[ ]:


# SI_NI_FGSM+MLP生成的对抗样本
adv_ex_data_SI_NI_FGSM_MLP = dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/adv_ex_features_SI_NI_FGSM_MLP.pkl",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                               
            target_transform=torch.tensor)

test_loader_SI_NI_FGSM_MLP = torch.utils.data.DataLoader(dataset=adv_ex_data_SI_NI_FGSM_MLP, batch_size=batch_size, shuffle=False)

# SI_NI_FGSM+DNN生成的对抗样本
adv_ex_data_SI_NI_FGSM_DNN = dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/adv_ex_features_SI_NI_FGSM_DNN.pkl",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                                
            target_transform=torch.tensor)
test_loader_SI_NI_FGSM_DNN = torch.utils.data.DataLoader(dataset=adv_ex_data_SI_NI_FGSM_DNN, batch_size=batch_size, shuffle=False)

# SI_NI_FGSM+LSTM生成的对抗样本
adv_ex_data_SI_NI_FGSM_LSTM = dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/adv_ex_features_SI_NI_FGSM_LSTM.pkl",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                                
            target_transform=torch.tensor)
test_loader_SI_NI_FGSM_LSTM = torch.utils.data.DataLoader(dataset=adv_ex_data_SI_NI_FGSM_LSTM, batch_size=batch_size, shuffle=False)


# ### 基于采样训练集生成的加强训练集的SI_NI_FGSM对抗样本

# In[ ]:


# SI_NI_FGSM+MLP
under_train_adv_ex_SI_NI_FGSM_MLP= dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/under_train_adv_ex_SI_NI_FGSM_MLP.pkl",
            target_file=under_train_labels_path,
            transform=torch.tensor,
            target_transform=torch.tensor
            )

test_loader_under_train_adv_ex_SI_NI_FGSM_MLP = torch.utils.data.DataLoader(dataset=under_train_adv_ex_SI_NI_FGSM_MLP, batch_size=batch_size, shuffle=False)

# SI_NI_FGSM+DNN
under_train_adv_ex_SI_NI_FGSM_DNN= dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/under_train_adv_ex_SI_NI_FGSM_DNN.pkl",
            target_file=under_train_labels_path,
            transform=torch.tensor,
            target_transform=torch.tensor
            )

test_loader_under_train_adv_ex_SI_NI_FGSM_DNN = torch.utils.data.DataLoader(dataset=under_train_adv_ex_SI_NI_FGSM_DNN, batch_size=batch_size, shuffle=False)

# SI_NI_FGSM+LSTM
under_train_adv_ex_SI_NI_FGSM_LSTM= dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/under_train_adv_ex_SI_NI_FGSM_LSTM.pkl",
            target_file=under_train_labels_path,
            transform=torch.tensor,
            target_transform=torch.tensor
            )

test_loader_under_train_adv_ex_SI_NI_FGSM_LSTM = torch.utils.data.DataLoader(dataset=under_train_adv_ex_SI_NI_FGSM_LSTM, batch_size=batch_size, shuffle=False)


# ### Autoattack生成的对抗样本

# In[ ]:


# Autoattack + 无目标 + MLP 生成的对抗样本
adv_ex_data_auto_no_target_MLP = dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/adv_ex_features_auto_no_target_MLP.pkl",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                               
            target_transform=torch.tensor)
test_loader_auto_no_target_MLP = torch.utils.data.DataLoader(dataset=adv_ex_data_auto_no_target_MLP, batch_size=batch_size, shuffle=False)

# Autoattack + 无目标 + DNN 生成的对抗样本
adv_ex_data_auto_no_target_DNN = dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/adv_ex_features_auto_no_target_DNN.pkl",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                                
            target_transform=torch.tensor)
test_loader_auto_no_target_DNN = torch.utils.data.DataLoader(dataset=adv_ex_data_auto_no_target_DNN, batch_size=batch_size, shuffle=False)

# Autoattack + 无目标 + LSTM 生成的对抗样本
adv_ex_data_auto_no_target_LSTM = dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/adv_ex_features_auto_no_target_LSTM.pkl",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                               
            target_transform=torch.tensor)
test_loader_auto_no_target_LSTM = torch.utils.data.DataLoader(dataset=adv_ex_data_auto_no_target_LSTM, batch_size=batch_size, shuffle=False)

# Autoattack + 有目标 + MLP 生成的对抗样本
adv_ex_data_auto_target_MLP = dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/adv_ex_features_auto_target_MLP.pkl",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                               
            target_transform=torch.tensor)
test_loader_auto_target_MLP = torch.utils.data.DataLoader(dataset=adv_ex_data_auto_target_MLP, batch_size=batch_size, shuffle=False)

# Autoattack + 有目标 + DNN 生成的对抗样本
adv_ex_data_auto_target_DNN = dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/adv_ex_features_auto_target_DNN.pkl",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                                
            target_transform=torch.tensor)
test_loader_auto_target_DNN = torch.utils.data.DataLoader(dataset=adv_ex_data_auto_target_DNN, batch_size=batch_size, shuffle=False)

# Autoattack + 有目标 + LSTM 生成的对抗样本
adv_ex_data_auto_target_LSTM = dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/adv_ex_features_auto_target_LSTM.pkl",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                                
            target_transform=torch.tensor)
test_loader_auto_target_LSTM = torch.utils.data.DataLoader(dataset=adv_ex_data_auto_target_LSTM, batch_size=batch_size, shuffle=False)


# ### UAP对抗样本 (UAP + 恶意流量 + MLP生成的对抗样本)

# In[ ]:


adv_ex_data_UAP1 = dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/test_malicious_UAP0.1_MLP.pkl",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                               
            target_transform=torch.tensor)
test_loader_uap_mlp1 = torch.utils.data.DataLoader(dataset=adv_ex_data_UAP1, batch_size=batch_size, shuffle=False)


adv_ex_data_UAP2 = dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/test_malicious_UAP0.2_MLP.pkl",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                                
            target_transform=torch.tensor)
test_loader_uap_mlp2 = torch.utils.data.DataLoader(dataset=adv_ex_data_UAP2, batch_size=batch_size, shuffle=False)


# ## 加载数据集

# In[ ]:


train_data,val_data,test_data = dataset.get_dataset(data_path = DATA_DIR, balanced = True)


# 拼接原始训练集和对抗样本，用于后续训练

# In[ ]:


# 拼接数据集，将under_train_adv_ex_SI_NI_FGSM_MLP, under_train_adv_ex_SI_NI_FGSM_DNN, under_train_adv_ex_SI_NI_FGSM_LSTM 新对抗样本一同添加至原始训练数据集
data_list = [train_data, under_train_adv_ex_SI_NI_FGSM_MLP, under_train_adv_ex_SI_NI_FGSM_DNN, under_train_adv_ex_SI_NI_FGSM_LSTM]

adv_ex_train_data = torch.utils.data.ConcatDataset(data_list)

batch_size = 64
train_loader = torch.utils.data.DataLoader(dataset=adv_ex_train_data, batch_size=batch_size, shuffle=True)

valid_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)


# ## 对抗训练MLP/DNN/LSTM模型

# In[ ]:


# 建立MLP
model = MLP(49, 64, 64, 8)

# # 建立DNN
# model = DNN()

# # 建立LSTM
# model = LSTM(49,64,8,3)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 优化器
learning_rate = 0.02
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Epochs 轮数
num_epochs = 10


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


# 保存训练好的模型

# In[ ]:


# 保存模型
# 新的MLP/DNN/LSTM模型
path = '../adversarial_train/adv_ex_train_models/SI_NI_FGSM_MLP_adv_model.pt'
# path = '../adversarial_train/adv_ex_train_models/SI_NI_FGSM_DNN_adv_model.pt'
# path = '../adversarial_train/adv_ex_train_models/SI_NI_FGSM_LSTM_adv_model.pt'

torch.save( model.state_dict(), path)


# In[ ]:


from sklearn.metrics import classification_report

labels = ['Benign', 'Botnet ARES', 'Brute Force','DDoS','DoS', 'Infiltration','PortScan','Web Attack']
print("Training Set -- Classification Report", end="\n\n")
print(classification_report(train_output_true, train_output_pred, target_names=labels))

print("Validation Set -- Classification Report", end="\n\n")
print(classification_report(valid_output_true, valid_output_pred, target_names=labels))


# ## 利用对抗模型分类对抗样本，观察准确率

# ### 分类FGSM攻击生成的对抗样本

# 原始模型对对抗样本的分类情况

# In[ ]:


from model.create_model import my_test

# 导入模型
model1 = MLP(49, 64, 64, 8)
model1.to(device)
model1.load_state_dict(torch.load('../model/create_model/created_models/MLP.pt'))

# model1 = DNN()
# model1.to(device)
# model1.load_state_dict(torch.load('../model/create_model/created_models/DNN.pt'))

# model1 = LSTM(49,64,8,3)
# model1.to(device)
# model1.load_state_dict(torch.load('../model/create_model/created_models/LSTM.pt'))

history = my_test.test(model1, criterion, test_loader_FGSM_MLP, device)

test_output_true = history['test']['output_true']
test_output_pred = history['test']['output_pred']
test_output_pred_prob = history['test']['output_pred_prob']


# 对抗模型分类FGSM攻击生成的对抗样本，准确率提升明显

# In[ ]:


# 导入新的模型
model2 = MLP(49, 64, 64, 8)
model2.to(device)
model2.load_state_dict(torch.load('../adversarial_train/adv_ex_train_models/SI_NI_FGSM_MLP_adv_model.pt'))

# model2 = DNN()
# model2.to(device)
# model2.load_state_dict(torch.load('../adversarial_train/adv_ex_train_models/SI_NI_FGSM_DNN_adv_model.pt'))

# model2 = LSTM(49,64,8,3)
# model2.to(device)
# model2.load_state_dict(torch.load('../adversarial_train/adv_ex_train_models/SI_NI_FGSM_LSTM_adv_model.pt'))

# 使用对抗训练后的新模型，重新分类对抗样本
history = my_test.test(model2, criterion, test_loader_FGSM_MLP, device)

test_output_true = history['test']['output_true']
test_output_pred = history['test']['output_pred']
test_output_pred_prob = history['test']['output_pred_prob']


# In[ ]:


labels = ['Benign', 'Botnet ARES', 'Brute Force','DDoS','DoS', 'Infiltration','PortScan','Web Attack']
print(history['test']['accuracy'])
print("Classification Report", end="\n\n")
print(classification_report(test_output_true, test_output_pred, target_names=labels))


# FGSM+DNN

# In[ ]:


# 原始模型分类准确率
history = my_test.test(model1, criterion, test_loader_FGSM_DNN, device)
# 对抗模型分类准确率
history = my_test.test(model2, criterion, test_loader_FGSM_DNN, device)


# FGSM+LSTM

# In[ ]:


# 原始模型分类准确率
history = my_test.test(model1, criterion, test_loader_FGSM_LSTM, device)
# 对抗模型分类准确率
history = my_test.test(model2, criterion, test_loader_FGSM_LSTM, device)


# ### 分类NI_FGSM攻击生成的对抗样本，准确率提升明显

# In[ ]:


# 原始模型分类准确率
history = my_test.test(model1, criterion, test_loader_NI_FGSM_MLP, device)
# 对抗模型分类准确率
history = my_test.test(model2, criterion, test_loader_NI_FGSM_MLP, device)

test_output_true = history['test']['output_true']
test_output_pred = history['test']['output_pred']
test_output_pred_prob = history['test']['output_pred_prob']
print(classification_report(test_output_true, test_output_pred, target_names=labels))


# NI-FGSM + DNN

# In[ ]:


# 原始模型分类准确率
history = my_test.test(model1, criterion, test_loader_NI_FGSM_DNN, device)
# 对抗模型分类准确率
history = my_test.test(model2, criterion, test_loader_NI_FGSM_DNN, device)


# NI-FGSM + LSTM

# In[ ]:


# 原始模型分类准确率
history = my_test.test(model1, criterion, test_loader_NI_FGSM_LSTM, device)
# 对抗模型分类准确率
history = my_test.test(model2, criterion, test_loader_NI_FGSM_LSTM, device)


# ### 分类SI_NI_FGSM攻击生成的对抗样本

# SI_NI_FGSM+MLP生成的对抗样本

# In[ ]:


history = my_test.test(model1, criterion, test_loader_SI_NI_FGSM_MLP, device)

history = my_test.test(model2, criterion, test_loader_SI_NI_FGSM_MLP, device)

test_output_true = history['test']['output_true']
test_output_pred = history['test']['output_pred']
test_output_pred_prob = history['test']['output_pred_prob']
print(classification_report(test_output_true, test_output_pred, target_names=labels))


# SI_NI_FGSM+DNN生成的对抗样本

# In[ ]:


history = my_test.test(model1, criterion, test_loader_SI_NI_FGSM_DNN, device)

history = my_test.test(model2, criterion, test_loader_SI_NI_FGSM_DNN, device)


# 分类SI_NI_FGSM+LSTM生成的对抗样本

# In[ ]:


history = my_test.test(model1, criterion, test_loader_SI_NI_FGSM_LSTM, device)

history = my_test.test(model2, criterion, test_loader_SI_NI_FGSM_LSTM, device)


# ### 分类原始测试数据集

# In[ ]:


test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

history = my_test.test(model1, criterion, test_loader, device)

history = my_test.test(model2, criterion, test_loader, device)


# ## 分类Autoattck攻击生成的对抗样本

# Autoattack + 无目标 + MLP 生成的对抗样本

# In[ ]:


history = my_test.test(model1, criterion, test_loader_auto_no_target_MLP, device)

history = my_test.test(model2, criterion, test_loader_auto_no_target_MLP, device)


# Autoattack + 无目标 + DNN 生成的对抗样本

# In[ ]:


history = my_test.test(model1, criterion, test_loader_auto_no_target_DNN, device)

history = my_test.test(model2, criterion, test_loader_auto_no_target_DNN, device)


# Autoattack + 无目标 + LSTM 生成的对抗样本

# In[ ]:


history = my_test.test(model1, criterion, test_loader_auto_no_target_LSTM, device)

history = my_test.test(model2, criterion, test_loader_auto_no_target_LSTM, device)


# Autoattack + 有目标 + MLP 生成的对抗样本

# In[ ]:


history = my_test.test(model1, criterion, test_loader_auto_target_MLP, device)

history = my_test.test(model2, criterion, test_loader_auto_target_MLP, device)


# Autoattack + 有目标 + DNN 生成的对抗样本

# In[ ]:


history = my_test.test(model1, criterion, test_loader_auto_target_DNN, device)

history = my_test.test(model2, criterion, test_loader_auto_target_DNN, device)


# Autoattack + 有目标 + LSTM 生成的对抗样本

# In[ ]:


history = my_test.test(model1, criterion, test_loader_auto_target_LSTM, device)

history = my_test.test(model2, criterion, test_loader_auto_target_LSTM, device)


# ### 由于Autoattack和SINIFGSM攻击方法可能存在相似之处，都是迭代梯度的攻击方式。  
# ### 所以出现了上述防御结果：基于SINIFGSM的对抗训练模型也能一定程度上防御Autoattack攻击

# -----------------------------------------

# ## 测试对抗模型对UAP攻击生成的对抗样本的防御效果

# In[ ]:


# 原始模型分类准确率
history = my_test.test(model1, criterion, test_loader_uap_mlp1, device)
# 对抗模型分类准确率
history = my_test.test(model2, criterion, test_loader_uap_mlp1, device)


# In[ ]:


test_output_true = history['test']['output_true']
test_output_pred = history['test']['output_pred']
test_output_pred_prob = history['test']['output_pred_prob']
print(classification_report(test_output_true, test_output_pred, target_names=labels))


# In[ ]:


history = my_test.test(model1, criterion, test_loader_uap_mlp2, device)

history = my_test.test(model2, criterion, test_loader_uap_mlp2, device)

