#!/usr/bin/env python
# coding: utf-8

# # 实施FGSM攻击(三种模型和测试集恶意流量生成对抗样本)
# ----------------------------------------------

# In[ ]:


import torch
import os
import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from attacks import FGSM
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


# ### 加载模型和数据

# In[ ]:


# 加载模型
model_MLP = MLP(49, 64, 64, 8)
model_MLP.to(device=device)
model_MLP.load_state_dict(torch.load('../model/create_model/created_models/MLP.pt'))

model_DNN = DNN()
model_DNN.to(device)
model_DNN.load_state_dict(torch.load('../model/create_model/created_models/DNN.pt'))

model_LSTM = LSTM(49,64,8,3)
model_LSTM.to(device=device)
model_LSTM.load_state_dict(torch.load('../model/create_model/created_models/LSTM.pt'))

#加载测试数据
_,_,test_data = dataset.get_dataset(data_path = DATA_DIR, balanced = True)

batch_size = 64
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# 损失函数
criterion = torch.nn.CrossEntropyLoss()

# 加载测试集数据里的恶意流量，用于生成对抗样本
test_malicious_features_path = f"{DATA_DIR}/processed/test/test_malicious_features.pkl"
test_malicious_labels_path = f"{DATA_DIR}/processed/test/test_malicious_labels.pkl"

test_malicious_data = dataset.CICIDSDataset(
            features_file=test_malicious_features_path,
            target_file=test_malicious_labels_path,
            transform=torch.tensor,
            target_transform=torch.tensor
            )

test_malicious_loader = torch.utils.data.DataLoader(dataset=test_malicious_data, batch_size=batch_size, shuffle=False)

# 模型对原始恶意流量的分类准确率
# history = my_test.test(model_MLP, criterion, test_malicious_loader, device)
# test_output_true = history['test']['output_true']
# test_output_pred = history['test']['output_pred']
# test_accuracy = history['test']['accuracy']


# 原始干净数据分类报告

# In[ ]:


# labels = ['Benign', 'Botnet ARES', 'Brute Force','DDoS','DoS', 'Infiltration','PortScan','Web Attack']
# print(classification_report(test_output_true, test_output_pred, target_names=labels))


# ## FGSM攻击

# ### FGSM攻击MLP模型

# In[ ]:


accuracies_adv = []
adv_examples = []
epsilons = [0,0.05,0.1,0.15,0.2]
criterion = torch.nn.CrossEntropyLoss()
# Run test for each epsilon
for eps in epsilons:
    acc, ex = FGSM.generate_adv_ex_fgsm(model_MLP, device, criterion, test_malicious_loader, eps)
    accuracies_adv.append(acc)
    if eps == 0.1:
        adv_examples.append(ex)


# In[ ]:


adv_examples = np.vstack(adv_examples[0])
columns = test_malicious_data.features.columns
adv_examples = pd.DataFrame(adv_examples,columns=columns)
adv_examples.to_pickle('adversarial_examples/adv_ex_final/adv_ex_features_FGSM_MLP.pkl')


# ### FGSM攻击DNN模型

# In[ ]:


accuracies_adv = []
adv_examples = []
epsilons = [0,0.05,0.1,0.15,0.2]
# epsilons = [0.1]
criterion = torch.nn.CrossEntropyLoss()
# # Run test for each epsilon
for eps in epsilons:
    acc, ex = FGSM.generate_adv_ex_fgsm(model_DNN, device, criterion, test_malicious_loader, eps)
    accuracies_adv.append(acc)
    if eps == 0.1:
        adv_examples.append(ex)


# In[ ]:


adv_examples = np.vstack(adv_examples[0])
columns = test_malicious_data.features.columns
adv_examples = pd.DataFrame(adv_examples,columns=columns)
adv_examples.to_pickle('adversarial_examples/adv_ex_final/adv_ex_features_FGSM_DNN.pkl')


# ### FGSM攻击LSTM模型

# In[ ]:


accuracies_adv = []
adv_examples = []
epsilons = [0,0.05,0.1,0.15,0.2]
# epsilons = [0.1]
criterion = torch.nn.CrossEntropyLoss()
# # Run test for each epsilon
for eps in epsilons:
    acc, ex = FGSM.generate_adv_ex_fgsm(model_LSTM, device, criterion, test_malicious_loader, eps)
    accuracies_adv.append(acc)
    if eps == 0.1:
        adv_examples.append(ex)


# In[ ]:


adv_examples = np.vstack(adv_examples[0])
columns = test_malicious_data.features.columns
adv_examples = pd.DataFrame(adv_examples,columns=columns)
adv_examples.to_pickle('adversarial_examples/adv_ex_final/adv_ex_features_FGSM_LSTM.pkl')


# ## 测试各模型对FGSM对抗样本的分类情况

# ### 生成的FGSM+MLP对抗样本攻击DNN，LSTM

# In[ ]:


DATA_DIR_ADV_EXAMPLES  = os.path.join(os.path.abspath(".."),'adversarial_attacks','adversarial_examples')
adv_ex_data_FGSM_MLP = dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/adv_ex_features_FGSM_MLP.pkl",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                                #转换成张量
            target_transform=torch.tensor)
test_adv_ex_loader = torch.utils.data.DataLoader(dataset=adv_ex_data_FGSM_MLP, batch_size=batch_size, shuffle=False)


# In[ ]:


# MLP模型分类准确率
history = my_test.test(model_MLP, criterion, test_adv_ex_loader, device)
# DNN模型分类准确率
history = my_test.test(model_DNN, criterion, test_adv_ex_loader, device)
# LSTM模型分类准确率
history = my_test.test(model_LSTM, criterion, test_adv_ex_loader, device)


# ### FGSM+DNN生成的对抗样本

# In[ ]:


DATA_DIR_ADV_EXAMPLES  = os.path.join(os.path.abspath(".."),'adversarial_attacks','adversarial_examples')
adv_ex_data_FGSM_DNN = dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/adv_ex_features_FGSM_DNN.pkl",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                                
            target_transform=torch.tensor)
test_adv_ex_loader = torch.utils.data.DataLoader(dataset=adv_ex_data_FGSM_DNN, batch_size=batch_size, shuffle=False)


# In[ ]:


# MLP模型分类准确率
history = my_test.test(model_MLP, criterion, test_adv_ex_loader, device)
# DNN模型分类准确率
history = my_test.test(model_DNN, criterion, test_adv_ex_loader, device)
# LSTM模型分类准确率
history = my_test.test(model_LSTM, criterion, test_adv_ex_loader, device)


# ### FGSM+LSTM生成的对抗样本

# In[ ]:


DATA_DIR_ADV_EXAMPLES  = os.path.join(os.path.abspath(".."),'adversarial_attacks','adversarial_examples')
adv_ex_data_FGSM_LSTM = dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/adv_ex_features_FGSM_LSTM.pkl",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                               
            target_transform=torch.tensor)
test_adv_ex_loader = torch.utils.data.DataLoader(dataset=adv_ex_data_FGSM_LSTM, batch_size=batch_size, shuffle=False)


# In[ ]:


# MLP模型分类准确率
history = my_test.test(model_MLP, criterion, test_adv_ex_loader, device)
# DNN模型分类准确率
history = my_test.test(model_DNN, criterion, test_adv_ex_loader, device)
# LSTM模型分类准确率
history = my_test.test(model_LSTM, criterion, test_adv_ex_loader, device)

