#!/usr/bin/env python
# coding: utf-8

# # SI_NI_FGSM对抗攻击MLP,DNN,LSTM
# # 对抗样本基于测试集的恶意流量生成
# 
# # 此外，基于采样的训练集，生成加强训练集的对抗样本，便于后期对抗训练

# In[ ]:


import torch
import os
import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from attacks import SI_NI_FGSM
from model.models import MLP,DNN,LSTM
from model.create_model import my_test,dataset,utils
from logger import logger
from sklearn.metrics import classification_report


# In[ ]:


LOG_CONFIG_PATH = os.path.join(os.path.abspath(".."), "logger", "logger_config.json")
LOG_DIR   = os.path.join(os.path.abspath(".."), "attack_logs")
DATA_DIR  = os.path.join(os.path.abspath(".."), "data")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

utils.mkdir(LOG_DIR)
logger.setup_logging(save_dir=LOG_DIR, log_config=LOG_CONFIG_PATH)


# ## 加载模型及数据

# 加载模型

# In[ ]:


model_MLP = MLP(49, 64, 64, 8)
model_MLP.to(device=device)
model_MLP.load_state_dict(torch.load('../model/create_model/created_models/MLP.pt'))

model_DNN = DNN()
model_DNN.to(device)
model_DNN.load_state_dict(torch.load('../model/create_model/created_models/DNN.pt'))

model_LSTM = LSTM(49,64,8,3)
model_LSTM.to(device=device)
model_LSTM.load_state_dict(torch.load('../model/create_model/created_models/LSTM.pt'))

#加载原始测试数据
_,_,test_data = dataset.get_dataset(data_path = DATA_DIR, balanced = True)

batch_size = 64
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# # 损失函数
criterion = torch.nn.CrossEntropyLoss()

# history = my_test.test(model, criterion, test_loader, device)
# test_output_true = history['test']['output_true']
# test_output_pred = history['test']['output_pred']
# test_accuracy = history['test']['accuracy']


# 加载test数据集中的恶意流量

# In[ ]:


test_malicious_features_path = f"{DATA_DIR}/processed/test/test_malicious_features.pkl"
test_malicious_labels_path = f"{DATA_DIR}/processed/test/test_malicious_labels.pkl"

test_malicious_data = dataset.CICIDSDataset(
            features_file=test_malicious_features_path,
            target_file=test_malicious_labels_path,
            transform=torch.tensor,
            target_transform=torch.tensor
            )

test_malicious_loader = torch.utils.data.DataLoader(dataset=test_malicious_data, batch_size=batch_size, shuffle=False)


# 加载test数据集中的良性流量

# In[ ]:


test_benign_features = pd.read_pickle(f"{DATA_DIR}/processed/test/test_benign_features.pkl")
test_benign_labels = pd.read_pickle(f"{DATA_DIR}/processed/test/test_benign_labels.pkl")
test_benign = pd.concat([test_benign_features,test_benign_labels],axis=1)


# 加载从训练集采样的样本

# In[ ]:


under_train_features_path = f"{DATA_DIR}/processed/train/under_sampler_train_features_balanced.pkl"
under_train_labels_path = f"{DATA_DIR}/processed/train/under_sampler_train_labels_balanced.pkl"

under_train_data = dataset.CICIDSDataset(
            features_file=under_train_features_path,
            target_file=under_train_labels_path,
            transform=torch.tensor,
            target_transform=torch.tensor
            )

under_train_loader = torch.utils.data.DataLoader(dataset=under_train_data, batch_size=batch_size, shuffle=False)


# ## SI_NI_FGSM攻击

# ### SI_NI_FGSM攻击MLP

# 查看原始恶意流量分类报告

# In[ ]:


history = my_test.test(model_MLP, criterion, test_malicious_loader, device)
test_output_true = history['test']['output_true']
test_output_pred = history['test']['output_pred']
test_accuracy = history['test']['accuracy']
print(test_accuracy)

# 攻击前的分类报告
labels = ['Benign', 'Botnet ARES', 'Brute Force','DDoS','DoS', 'Infiltration','PortScan','Web Attack']
print(classification_report(test_output_true, test_output_pred, target_names=labels))


# 利用恶意流量生成对抗样本

# In[ ]:


# 攻击
epsilons = 0.1
acc, ex = SI_NI_FGSM.generate_adv_ex_sinifgsm(model_MLP, device, test_malicious_loader, epsilons)


# In[ ]:


adv_examples = np.vstack(ex)
# 保存对抗样本
adv_examples = pd.DataFrame(adv_examples,columns=test_malicious_data.features.columns)           
# export_adv = pd.concat([adv_examples,test_malicious_data.labels],axis=1)
# export_adv.to_csv("adversarial_examples/adv_ex_final/adversarial_examples_SI_NI_FGSM_MLP.csv") # 对抗样本导入在一个.csv文件里
adv_examples.to_pickle('adversarial_examples/adv_ex_final/adv_ex_features_SI_NI_FGSM_MLP.pkl')


# ### 利用采样的训练数据生成加强的对抗样本-MLP

# In[ ]:


# 攻击
epsilons = 0.1
acc, ex = SI_NI_FGSM.generate_adv_ex_sinifgsm(model_MLP, device, under_train_loader, epsilons)

# 保存
adv_examples = np.vstack(ex)
adv_examples = pd.DataFrame(adv_examples,columns=under_train_data.features.columns) 
adv_examples.to_pickle('adversarial_examples/adv_ex_final/under_train_adv_ex_SI_NI_FGSM_MLP.pkl')


# ### SI_NI_FGSM攻击DNN

# In[ ]:


# DNN对原始测试数据的分类准确率
history = my_test.test(model_DNN, criterion, test_malicious_loader, device)
test_output_true = history['test']['output_true']
test_output_pred = history['test']['output_pred']
test_accuracy = history['test']['accuracy']
print(test_accuracy)
# 攻击
epsilons = 0.1
acc, ex = SI_NI_FGSM.generate_adv_ex_sinifgsm(model_DNN, device, test_malicious_loader, epsilons)

adv_examples = np.vstack(ex)
adv_examples.to_pickle('adversarial_examples/adv_ex_final/adv_ex_features_SI_NI_FGSM_DNN.pkl')


# ### 利用采样的训练数据生成加强的对抗样本-DNN

# In[ ]:


# 攻击
epsilons = 0.1
acc, ex = SI_NI_FGSM.generate_adv_ex_sinifgsm(model_DNN, device, under_train_loader, epsilons)
# 保存
adv_examples = np.vstack(ex)
adv_examples = pd.DataFrame(adv_examples,columns=under_train_data.features.columns) 
adv_examples.to_pickle('adversarial_examples/adv_ex_final/under_train_adv_ex_SI_NI_FGSM_DNN.pkl')


# ### SI_NI_FGSM攻击LSTM

# In[ ]:


# LSTM对原始测试恶意流量数据的分类准确率
history = my_test.test(model_LSTM, criterion, test_malicious_loader, device)
test_output_true = history['test']['output_true']
test_output_pred = history['test']['output_pred']
test_accuracy = history['test']['accuracy']
print(test_accuracy)

# 攻击
epsilons = 0.1
model_LSTM.train()
acc, ex = SI_NI_FGSM.generate_adv_ex_sinifgsm(model_LSTM, device, test_malicious_loader, epsilons)
adv_examples = np.vstack(ex)
adv_examples.to_pickle('adversarial_examples/adv_ex_final/adv_ex_features_SI_NI_FGSM_LSTM.pkl')


# ### 利用采样的训练数据生成加强的对抗样本-LSTM

# In[ ]:


# 攻击
epsilons = 0.1
acc, ex = SI_NI_FGSM.generate_adv_ex_sinifgsm(model_LSTM, device, under_train_loader, epsilons)

# 保存
adv_examples = np.vstack(ex)
adv_examples = pd.DataFrame(adv_examples,columns=under_train_data.features.columns) 
adv_examples.to_pickle('adversarial_examples/adv_ex_final/under_train_adv_ex_SI_NI_FGSM_LSTM.pkl')

