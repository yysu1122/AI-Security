#!/usr/bin/env python
# coding: utf-8

# # NI-FGSM对抗攻击
# --------------------------------------------------

# In[ ]:


import torch
import os
import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from attacks import NI_FGSM
from model.models import MLP,DNN,LSTM
from model.create_model import my_test,dataset,utils
from logger import logger


# In[ ]:


LOG_CONFIG_PATH = os.path.join(os.path.abspath(".."), "logger", "logger_config.json")
LOG_DIR  = os.path.join(os.path.abspath(".."), "attack_logs")
DATA_DIR = os.path.join(os.path.abspath(".."), "data")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

utils.mkdir(LOG_DIR)
logger.setup_logging(save_dir=LOG_DIR, log_config=LOG_CONFIG_PATH)


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

#加载测试数据
_,_,test_data = dataset.get_dataset(data_path = DATA_DIR, balanced = True)

batch_size = 64
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# 损失函数
criterion = torch.nn.CrossEntropyLoss()


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


# 原始干净数据分类报告

# In[ ]:


from sklearn.metrics import classification_report

# 原始恶意流量的分类情况
history = my_test.test(model_MLP, criterion, test_malicious_loader, device)
test_output_true = history['test']['output_true']
test_output_pred = history['test']['output_pred']
test_accuracy = history['test']['accuracy']

labels = ['Benign', 'Botnet ARES', 'Brute Force','DDoS','DoS', 'Infiltration','PortScan','Web Attack']
print(classification_report(test_output_true, test_output_pred, target_names=labels))


# In[ ]:


history = my_test.test(model_DNN, criterion, test_malicious_loader, device)
history = my_test.test(model_LSTM, criterion, test_malicious_loader, device)


# ## NI_FGSM攻击

# ### NI-FGSM攻击MLP 

# In[ ]:


adv_examples = []
epsilons = 0.1
acc, ex = NI_FGSM.generate_adv_ex_nifgsm(model_MLP, device, test_malicious_loader, epsilons)
adv_examples.append(ex)


# In[ ]:


adv_examples = np.vstack(adv_examples[0])
# 保存对抗样本
columns = test_malicious_data.features.columns
adv_examples = pd.DataFrame(adv_examples,columns=columns)           
# export_adv = pd.concat([adv_examples,y_test],axis=1)
# export_adv.to_csv("adversarial_examples/adversarial_examples_NI_FGSM.csv")
adv_examples.to_pickle('adversarial_examples/adv_ex_final/adv_ex_features_NI_FGSM_MLP.pkl')


# ### NI-FGSM攻击DNN 

# In[ ]:


adv_examples = []
epsilons = 0.1
acc, ex = NI_FGSM.generate_adv_ex_nifgsm(model_DNN, device, test_malicious_loader, epsilons)
adv_examples.append(ex)


# In[ ]:


adv_examples = np.vstack(adv_examples[0])
adv_examples = pd.DataFrame(adv_examples,columns=columns)           
adv_examples.to_pickle('adversarial_examples/adv_ex_final/adv_ex_features_NI_FGSM_DNN.pkl')


# ### NI-FGSM攻击LSTM

# In[ ]:


# accuracies_adv = []
adv_examples = []
epsilons = 0.1
model_LSTM.train()
acc, ex = NI_FGSM.generate_adv_ex_nifgsm(model_LSTM, device, test_malicious_loader, epsilons)
adv_examples.append(ex)


# In[ ]:


adv_examples = np.vstack(adv_examples[0])
adv_examples = pd.DataFrame(adv_examples,columns=columns)           
adv_examples.to_pickle('adversarial_examples/adv_ex_final/adv_ex_features_NI_FGSM_LSTM.pkl')


# ## 测试各模型对NI-FGSM对抗样本的分类情况

# ### 生成的NI-FGSM + MLP对抗样本攻击模型

# In[ ]:


DATA_DIR_ADV_EXAMPLES  = os.path.join(os.path.abspath(".."),'adversarial_attacks','adversarial_examples')
adv_ex_data_NI_FGSM_MLP = dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/adv_ex_features_NI_FGSM_MLP.pkl",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                               
            target_transform=torch.tensor)
test_adv_ex_loader = torch.utils.data.DataLoader(dataset=adv_ex_data_NI_FGSM_MLP, batch_size=batch_size, shuffle=False)


# In[ ]:


# MLP模型分类准确率
history = my_test.test(model_MLP, criterion, test_adv_ex_loader, device)
# DNN模型分类准确率
history = my_test.test(model_DNN, criterion, test_adv_ex_loader, device)
# LSTM模型分类准确率
history = my_test.test(model_LSTM, criterion, test_adv_ex_loader, device)


# ### 生成的NI-FGSM + DNN对抗样本攻击模型

# In[ ]:


DATA_DIR_ADV_EXAMPLES  = os.path.join(os.path.abspath(".."),'adversarial_attacks','adversarial_examples')
adv_ex_data_NI_FGSM_DNN = dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/adv_ex_features_NI_FGSM_DNN.pkl",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                               
            target_transform=torch.tensor)
test_adv_ex_loader = torch.utils.data.DataLoader(dataset=adv_ex_data_NI_FGSM_DNN, batch_size=batch_size, shuffle=False)


# In[ ]:


# MLP模型分类准确率
history = my_test.test(model_MLP, criterion, test_adv_ex_loader, device)
# DNN模型分类准确率
history = my_test.test(model_DNN, criterion, test_adv_ex_loader, device)
# LSTM模型分类准确率
history = my_test.test(model_LSTM, criterion, test_adv_ex_loader, device)


# ### 生成的NI-FGSM + LSTM对抗样本攻击模型

# In[ ]:


DATA_DIR_ADV_EXAMPLES  = os.path.join(os.path.abspath(".."),'adversarial_attacks','adversarial_examples')
adv_ex_data_NI_FGSM_LSTM = dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/adv_ex_features_NI_FGSM_LSTM.pkl",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                              
            target_transform=torch.tensor)
test_adv_ex_loader = torch.utils.data.DataLoader(dataset=adv_ex_data_NI_FGSM_LSTM, batch_size=batch_size, shuffle=False)


# In[ ]:


# MLP模型分类准确率
history = my_test.test(model_MLP, criterion, test_adv_ex_loader, device)
# DNN模型分类准确率
history = my_test.test(model_DNN, criterion, test_adv_ex_loader, device)
# LSTM模型分类准确率
history = my_test.test(model_LSTM, criterion, test_adv_ex_loader, device)

