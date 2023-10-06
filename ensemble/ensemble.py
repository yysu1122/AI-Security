#!/usr/bin/env python
# coding: utf-8

# # 将模型进行集成，得到综合预测结果

# In[ ]:


import numpy as np
import torch
import sys
sys.path.append('..')
from model.models import MLP,DNN,LSTM
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from model.create_model import dataset
import os
from model.create_model import my_test
from tqdm import tqdm


# ## 加载模型

# In[ ]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
# 加载模型
model_MLP = MLP(49,64,64,8)
model_MLP.load_state_dict(torch.load('../model/create_model/created_models/MLP.pt'))
model_MLP.to(device)
model_DNN = DNN()
model_DNN.load_state_dict(torch.load('../model/create_model/created_models/DNN.pt'))
model_DNN.to(device)
model_LSTM = LSTM(49,64,8,3)
model_LSTM.load_state_dict(torch.load('../model/create_model/created_models/LSTM.pt'))
model_LSTM.to(device)


# ## 加载数据

# In[ ]:


DATA_DIR_ADV_EXAMPLES  = os.path.join(os.path.abspath(".."),'adversarial_attacks','adversarial_examples')
DATA_DIR  = os.path.join(os.path.abspath(".."), "data")
print(DATA_DIR_ADV_EXAMPLES)

# 损失函数
criterion = nn.CrossEntropyLoss()
batch_size = 64
train_data, val_data, test_data = dataset.get_dataset(data_path=DATA_DIR, balanced=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


# 加载test数据集中的良性流量

# In[ ]:


test_benign_features_path = f"{DATA_DIR}/processed/test/test_benign_features.pkl"
test_benign_labels_path = f"{DATA_DIR}/processed/test/test_benign_labels.pkl"

test_benign_data = dataset.CICIDSDataset(
            features_file=test_benign_features_path,
            target_file=test_benign_labels_path,
            transform=torch.tensor,
            target_transform=torch.tensor
            )

test_benign_loader = torch.utils.data.DataLoader(dataset=test_benign_data, batch_size=batch_size, shuffle=False)


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


# ### FGSM 对抗样本

# In[ ]:


# FGSM + MLP生成的对抗样本
adv_ex_data_FGSM_MLP = dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/adv_ex_features_FGSM_MLP.pkl",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                                
            target_transform=torch.tensor)

test_loader_FGSM_MLP = torch.utils.data.DataLoader(dataset=adv_ex_data_FGSM_MLP, batch_size=batch_size, shuffle=False)

# FGSM + DNN生成的对抗样本
adv_ex_data_FGSM_DNN = dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/adv_ex_features_FGSM_DNN.pkl",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                                
            target_transform=torch.tensor)

test_loader_FGSM_DNN = torch.utils.data.DataLoader(dataset=adv_ex_data_FGSM_DNN, batch_size=batch_size, shuffle=False)

# FGSM + LSTM生成的对抗样本
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


# ### SI_NI_FGSM对抗样本

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


# ### Autoattack对抗样本

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


# In[ ]:


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


# ### UAP对抗样本

# In[ ]:


# UAP + 恶意流量 + MLP生成的对抗样本
adv_ex_data_UAP1 = dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/test_malicious_UAP0.1_MLP",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                                 
            target_transform=torch.tensor)
test_loader_uap_mlp1 = torch.utils.data.DataLoader(dataset=adv_ex_data_UAP1, batch_size=batch_size, shuffle=False)

adv_ex_data_UAP2 = dataset.CICIDSDataset(
            features_file=f"{DATA_DIR_ADV_EXAMPLES}/adv_ex_final/test_malicious_UAP0.2_MLP",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,                                 
            target_transform=torch.tensor)
test_loader_uap_mlp2 = torch.utils.data.DataLoader(dataset=adv_ex_data_UAP2, batch_size=batch_size, shuffle=False)


# -------------------------------  
# -------------------------------
# ## 利用投票机制集成模型

# ### 定义集成模型
# 方案1：按照三种模型的outputs输出最大值，预测样本标签  
# 方案2：按照三种模型的outputs输出均值，预测样本标签  
# 方案3：按照三种模型的输出标签，求得众数，输出最终预测标签 (投票最多的类别就是最终的预测类别)  
# 方案4：给每个模型设置权重，区别模型重要度（实验无重要度区分，未实现此方案）

# In[ ]:


#方案1
# 按照最大outputs判断预测标签  

class Ensemble_max(nn.Module):
    def __init__(self, model_list):
        super().__init__()
        self.model_list = nn.ModuleList(model_list)
        
    def forward(self, test_loader: torch.utils.data.DataLoader, device: torch.device):
        y_pred_label = []  # 预测标签
        output_true = []   # 真实标签
        with torch.no_grad():
            for (inputs, labels) in tqdm(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.squeeze(1)
                outputs_model = [] 
                for model in self.model_list:
                    model.eval() 
                    outputs = model(inputs)
                    outputs_model.append(outputs.cpu().tolist())

                outputs_model = torch.FloatTensor(outputs_model)
            
                y_pred_ensemble_temp = torch.max(outputs_model, dim=0)[0] # 求出三个模型输出置信度的最大值
                y_pred_label_temp = torch.argmax(y_pred_ensemble_temp, dim=1) # 求预测标签
            
                y_pred_label += y_pred_label_temp.tolist()
                output_true += labels.tolist()
         
        return y_pred_label, output_true


# In[ ]:


# 按照outputs的均值判断预测标签
from tqdm import tqdm

class Ensemble_mean(nn.Module):
    def __init__(self, model_list):
        super().__init__()
        self.model_list = nn.ModuleList(model_list)
        
    def forward(self, test_loader: torch.utils.data.DataLoader, device: torch.device):
        y_pred_label = []
        output_true = []
        with torch.no_grad():
            for (inputs, labels) in tqdm(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.squeeze(1)
                outputs_model = [] 
                for model in self.model_list:
                    model.eval() 
                    outputs = model(inputs)
                    outputs_model.append(outputs.cpu().tolist())

                outputs_model = torch.FloatTensor(outputs_model)
                y_pred_ensemble_temp = torch.mean(outputs_model, dim=0)  # 求出三个模型输出置信度的均值
                y_pred_label_temp = torch.argmax(y_pred_ensemble_temp, dim=1)  # 再根据最终均值Output求预测标签

                y_pred_label += y_pred_label_temp.tolist()
                output_true += labels.tolist()
       
        return y_pred_label, output_true


# In[ ]:


# 按照预测标签的众数判断最终预测标签
from tqdm import tqdm

class Ensemble_mode(nn.Module):
    def __init__(self, model_list):
        super().__init__()
        self.model_list = nn.ModuleList(model_list)
        
    def forward(self, test_loader: torch.utils.data.DataLoader, device: torch.device):
        y_pred_label = []
        output_true = []
        with torch.no_grad():
            for (inputs, labels) in tqdm(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.squeeze(1)
                outputs_model = [] 
                for model in self.model_list:
                    model.eval() 
                    outputs = model(inputs)
                    outputs_model.append(outputs.cpu().tolist())

                outputs_model = torch.FloatTensor(outputs_model)
        
                y_pred_label_temp = torch.argmax(outputs_model, dim=2)  # 先预测标签
                y_pred_ensemble_temp = torch.mode(y_pred_label_temp, dim=0)[0]  # 求得标签里的众数
             
                y_pred_label += y_pred_ensemble_temp.tolist()
                output_true += labels.tolist()

        return y_pred_label, output_true


# ### 构建集成模型

# In[ ]:


ensemble_max = Ensemble_max([model_MLP, model_DNN, model_LSTM])
ensemble_mean = Ensemble_mean([model_MLP, model_DNN, model_LSTM])
ensemble_mode = Ensemble_mode([model_MLP, model_DNN, model_LSTM])


# -----------------------------  
# -----------------------------
# ## 预测对抗样本

# FGSM+MLP生成的对抗样本

# In[ ]:


print('FGSM+MLP生成的对抗样本')
pred_label_max, true_label_max = ensemble_max(test_loader_FGSM_MLP, device)
acc_max = torch.sum(torch.LongTensor(pred_label_max) == torch.LongTensor(true_label_max)) / len(true_label_max)
print('集成模型(max)准确率: {:.2f}%'.format(acc_max * 100)) # 

pred_label_mean, true_label_mean = ensemble_mean(test_loader_FGSM_MLP, device)
acc_mean = torch.sum(torch.LongTensor(pred_label_mean) == torch.LongTensor(true_label_mean)) / len(true_label_mean)
print('集成模型(mean)准确率: {:.2f}%'.format(acc_mean * 100)) # 

pred_label_mode, true_label_mode = ensemble_mode(test_loader_FGSM_MLP, device)
acc_mode = torch.sum(torch.LongTensor(pred_label_mode) == torch.LongTensor(true_label_mode)) / len(true_label_mode)
print('集成模型(mode)准确率: {:.2f}%'.format(acc_mode * 100)) # 

# 原模型预测准确率
MLP_history = my_test.test(model_MLP, criterion, test_loader_FGSM_MLP, device)
DNN_history = my_test.test(model_DNN, criterion, test_loader_FGSM_MLP, device)
LSTM_history = my_test.test(model_LSTM, criterion, test_loader_FGSM_MLP, device)
print(MLP_history['test']['accuracy'],DNN_history['test']['accuracy'],LSTM_history['test']['accuracy'])


# FGSM+DNN生成的对抗样本

# In[ ]:


print('FGSM+DNN生成的对抗样本')
pred_label_max, true_label_max = ensemble_max(test_loader_FGSM_DNN, device)
acc_max = torch.sum(torch.LongTensor(pred_label_max) == torch.LongTensor(true_label_max)) / len(true_label_max)
print('集成模型(max)准确率: {:.2f}%'.format(acc_max * 100)) # 

pred_label_mean, true_label_mean = ensemble_mean(test_loader_FGSM_DNN, device)
acc_mean = torch.sum(torch.LongTensor(pred_label_mean) == torch.LongTensor(true_label_mean)) / len(true_label_mean)
print('集成模型(mean)准确率: {:.2f}%'.format(acc_mean * 100)) # 

pred_label_mode, true_label_mode = ensemble_mode(test_loader_FGSM_DNN, device)
acc_mode = torch.sum(torch.LongTensor(pred_label_mode) == torch.LongTensor(true_label_mode)) / len(true_label_mode)
print('集成模型(mode)准确率: {:.2f}%'.format(acc_mode * 100)) # 

# 原模型预测准确率
MLP_history = my_test.test(model_MLP, criterion, test_loader_FGSM_DNN, device)
DNN_history = my_test.test(model_DNN, criterion, test_loader_FGSM_DNN, device)
LSTM_history = my_test.test(model_LSTM, criterion, test_loader_FGSM_DNN, device)
print(MLP_history['test']['accuracy'],DNN_history['test']['accuracy'],LSTM_history['test']['accuracy'])


# FGSM+LSTM生成的对抗样本

# In[ ]:


print('FGSM+LSTM生成的对抗样本')
pred_label_max, true_label_max = ensemble_max(test_loader_FGSM_LSTM, device)
acc_max = torch.sum(torch.LongTensor(pred_label_max) == torch.LongTensor(true_label_max)) / len(true_label_max)
print('集成模型(max)准确率: {:.2f}%'.format(acc_max * 100)) # 

pred_label_mean, true_label_mean = ensemble_mean(test_loader_FGSM_LSTM, device)
acc_mean = torch.sum(torch.LongTensor(pred_label_mean) == torch.LongTensor(true_label_mean)) / len(true_label_mean)
print('集成模型(mean)准确率: {:.2f}%'.format(acc_mean * 100)) # 

pred_label_mode, true_label_mode = ensemble_mode(test_loader_FGSM_LSTM, device)
acc_mode = torch.sum(torch.LongTensor(pred_label_mode) == torch.LongTensor(true_label_mode)) / len(true_label_mode)
print('集成模型(mode)准确率: {:.2f}%'.format(acc_mode * 100)) # 

# 原模型预测准确率
MLP_history = my_test.test(model_MLP, criterion, test_loader_FGSM_LSTM, device)
DNN_history = my_test.test(model_DNN, criterion, test_loader_FGSM_LSTM, device)
LSTM_history = my_test.test(model_LSTM, criterion, test_loader_FGSM_LSTM, device)
print(MLP_history['test']['accuracy'],DNN_history['test']['accuracy'],LSTM_history['test']['accuracy'])


# NI_FGSM+MLP生成的对抗样本

# In[ ]:


print('NI_FGSM+MLP生成的对抗样本')
pred_label_max, true_label_max = ensemble_max(test_loader_NI_FGSM_MLP, device)
acc_max = torch.sum(torch.LongTensor(pred_label_max) == torch.LongTensor(true_label_max)) / len(true_label_max)
print('集成模型(max)准确率: {:.2f}%'.format(acc_max * 100)) # 

pred_label_mean, true_label_mean = ensemble_mean(test_loader_NI_FGSM_MLP, device)
acc_mean = torch.sum(torch.LongTensor(pred_label_mean) == torch.LongTensor(true_label_mean)) / len(true_label_mean)
print('集成模型(mean)准确率: {:.2f}%'.format(acc_mean * 100)) # 

pred_label_mode, true_label_mode = ensemble_mode(test_loader_NI_FGSM_MLP, device)
acc_mode = torch.sum(torch.LongTensor(pred_label_mode) == torch.LongTensor(true_label_mode)) / len(true_label_mode)
print('集成模型(mode)准确率: {:.2f}%'.format(acc_mode * 100)) # 

# 原模型预测准确率
MLP_history = my_test.test(model_MLP, criterion, test_loader_NI_FGSM_MLP, device)
DNN_history = my_test.test(model_DNN, criterion, test_loader_NI_FGSM_MLP, device)
LSTM_history = my_test.test(model_LSTM, criterion, test_loader_NI_FGSM_MLP, device)
print(MLP_history['test']['accuracy'],DNN_history['test']['accuracy'],LSTM_history['test']['accuracy'])


# NI_FGSM+DNN生成的对抗样本

# In[ ]:


print('NI_FGSM+DNN生成的对抗样本')
pred_label_max, true_label_max = ensemble_max(test_loader_NI_FGSM_DNN, device)
acc_max = torch.sum(torch.LongTensor(pred_label_max) == torch.LongTensor(true_label_max)) / len(true_label_max)
print('集成模型(max)准确率: {:.2f}%'.format(acc_max * 100)) # 

pred_label_mean, true_label_mean = ensemble_mean(test_loader_NI_FGSM_DNN, device)
acc_mean = torch.sum(torch.LongTensor(pred_label_mean) == torch.LongTensor(true_label_mean)) / len(true_label_mean)
print('集成模型(mean)准确率: {:.2f}%'.format(acc_mean * 100)) # 

pred_label_mode, true_label_mode = ensemble_mode(test_loader_NI_FGSM_DNN, device)
acc_mode = torch.sum(torch.LongTensor(pred_label_mode) == torch.LongTensor(true_label_mode)) / len(true_label_mode)
print('集成模型(mode)准确率: {:.2f}%'.format(acc_mode * 100)) # 

# 原模型预测准确率
MLP_history = my_test.test(model_MLP, criterion, test_loader_NI_FGSM_DNN, device)
DNN_history = my_test.test(model_DNN, criterion, test_loader_NI_FGSM_DNN, device)
LSTM_history = my_test.test(model_LSTM, criterion, test_loader_NI_FGSM_DNN, device)
print(MLP_history['test']['accuracy'],DNN_history['test']['accuracy'],LSTM_history['test']['accuracy'])


# NI_FGSM+LSTM生成的对抗样本

# In[ ]:


print('NI_FGSM+LSTM生成的对抗样本')
pred_label_max, true_label_max = ensemble_max(test_loader_NI_FGSM_LSTM, device)
acc_max = torch.sum(torch.LongTensor(pred_label_max) == torch.LongTensor(true_label_max)) / len(true_label_max)
print('集成模型(max)准确率: {:.2f}%'.format(acc_max * 100)) # 

pred_label_mean, true_label_mean = ensemble_mean(test_loader_NI_FGSM_LSTM, device)
acc_mean = torch.sum(torch.LongTensor(pred_label_mean) == torch.LongTensor(true_label_mean)) / len(true_label_mean)
print('集成模型(mean)准确率: {:.2f}%'.format(acc_mean * 100)) # 

pred_label_mode, true_label_mode = ensemble_mode(test_loader_NI_FGSM_LSTM, device)
acc_mode = torch.sum(torch.LongTensor(pred_label_mode) == torch.LongTensor(true_label_mode)) / len(true_label_mode)
print('集成模型(mode)准确率: {:.2f}%'.format(acc_mode * 100)) # 

# 原模型预测准确率
MLP_history = my_test.test(model_MLP, criterion, test_loader_NI_FGSM_LSTM, device)
DNN_history = my_test.test(model_DNN, criterion, test_loader_NI_FGSM_LSTM, device)
LSTM_history = my_test.test(model_LSTM, criterion, test_loader_NI_FGSM_LSTM, device)
print(MLP_history['test']['accuracy'],DNN_history['test']['accuracy'],LSTM_history['test']['accuracy'])


# SI_NI_FGSM+MLP生成的对抗样本

# In[ ]:


print('SI_NI_FGSM+MLP生成的对抗样本')
pred_label_max, true_label_max = ensemble_max(test_loader_SI_NI_FGSM_MLP, device)
acc_max = torch.sum(torch.LongTensor(pred_label_max) == torch.LongTensor(true_label_max)) / len(true_label_max)
print('集成模型(max)准确率: {:.2f}%'.format(acc_max * 100)) # 

pred_label_mean, true_label_mean = ensemble_mean(test_loader_SI_NI_FGSM_MLP, device)
acc_mean = torch.sum(torch.LongTensor(pred_label_mean) == torch.LongTensor(true_label_mean)) / len(true_label_mean)
print('集成模型(mean)准确率: {:.2f}%'.format(acc_mean * 100)) # 

pred_label_mode, true_label_mode = ensemble_mode(test_loader_SI_NI_FGSM_MLP, device)
acc_mode = torch.sum(torch.LongTensor(pred_label_mode) == torch.LongTensor(true_label_mode)) / len(true_label_mode)
print('集成模型(mode)准确率: {:.2f}%'.format(acc_mode * 100)) # 

# 原模型预测准确率
MLP_history = my_test.test(model_MLP, criterion, test_loader_SI_NI_FGSM_MLP, device)
DNN_history = my_test.test(model_DNN, criterion, test_loader_SI_NI_FGSM_MLP, device)
LSTM_history = my_test.test(model_LSTM, criterion, test_loader_SI_NI_FGSM_MLP, device)
print(MLP_history['test']['accuracy'],DNN_history['test']['accuracy'],LSTM_history['test']['accuracy'])


# SI_NI_FGSM+DNN生成的对抗样本

# In[ ]:


print('SI_NI_FGSM+DNN生成的对抗样本')
pred_label_max, true_label_max = ensemble_max(test_loader_SI_NI_FGSM_DNN, device)
acc_max = torch.sum(torch.LongTensor(pred_label_max) == torch.LongTensor(true_label_max)) / len(true_label_max)
print('集成模型(max)准确率: {:.2f}%'.format(acc_max * 100))  # 

pred_label_mean, true_label_mean = ensemble_mean(test_loader_SI_NI_FGSM_DNN, device)
acc_mean = torch.sum(torch.LongTensor(pred_label_mean) == torch.LongTensor(true_label_mean)) / len(true_label_mean)
print('集成模型(mean)准确率: {:.2f}%'.format(acc_mean * 100))  # 

pred_label_mode, true_label_mode = ensemble_mode(test_loader_SI_NI_FGSM_DNN, device)
acc_mode = torch.sum(torch.LongTensor(pred_label_mode) == torch.LongTensor(true_label_mode)) / len(true_label_mode)
print('集成模型(mode)准确率: {:.2f}%'.format(acc_mode * 100))  # 

# 原模型预测准确率
MLP_history = my_test.test(model_MLP, criterion, test_loader_SI_NI_FGSM_DNN, device)
DNN_history = my_test.test(model_DNN, criterion, test_loader_SI_NI_FGSM_DNN, device)
LSTM_history = my_test.test(model_LSTM, criterion, test_loader_SI_NI_FGSM_DNN, device)
print(MLP_history['test']['accuracy'],DNN_history['test']['accuracy'],LSTM_history['test']['accuracy'])


# SI_NI_FGSM+LSTM生成的对抗样本

# In[ ]:


print('SI_NI_FGSM+LSTM生成的对抗样本')
pred_label_max, true_label_max = ensemble_max(test_loader_SI_NI_FGSM_LSTM, device)
acc_max = torch.sum(torch.LongTensor(pred_label_max) == torch.LongTensor(true_label_max)) / len(true_label_max)
print('集成模型(max)准确率: {:.2f}%'.format(acc_max * 100))  # 

pred_label_mean, true_label_mean = ensemble_mean(test_loader_SI_NI_FGSM_LSTM, device)
acc_mean = torch.sum(torch.LongTensor(pred_label_mean) == torch.LongTensor(true_label_mean)) / len(true_label_mean)
print('集成模型(mean)准确率: {:.2f}%'.format(acc_mean * 100))  # 

pred_label_mode, true_label_mode = ensemble_mode(test_loader_SI_NI_FGSM_LSTM, device)
acc_mode = torch.sum(torch.LongTensor(pred_label_mode) == torch.LongTensor(true_label_mode)) / len(true_label_mode)
print('集成模型(mode)准确率: {:.2f}%'.format(acc_mode * 100))  # 

# 原模型预测准确率
MLP_history = my_test.test(model_MLP, criterion, test_loader_SI_NI_FGSM_LSTM, device)
DNN_history = my_test.test(model_DNN, criterion, test_loader_SI_NI_FGSM_LSTM, device)
LSTM_history = my_test.test(model_LSTM, criterion, test_loader_SI_NI_FGSM_LSTM, device)
print(MLP_history['test']['accuracy'],DNN_history['test']['accuracy'],LSTM_history['test']['accuracy'])


# Autoattack + 无目标 + MLP 生成的对抗样本

# In[ ]:


print('Autoattack + 无目标 + MLP 生成的对抗样本')
pred_label_max, true_label_max = ensemble_max(test_loader_auto_no_target_MLP, device)
acc_max = torch.sum(torch.LongTensor(pred_label_max) == torch.LongTensor(true_label_max)) / len(true_label_max)
print('集成模型(max)准确率: {:.2f}%'.format(acc_max * 100)) # 

pred_label_mean, true_label_mean = ensemble_mean(test_loader_auto_no_target_MLP, device)
acc_mean = torch.sum(torch.LongTensor(pred_label_mean) == torch.LongTensor(true_label_mean)) / len(true_label_mean)
print('集成模型(mean)准确率: {:.2f}%'.format(acc_mean * 100)) # 

pred_label_mode, true_label_mode = ensemble_mode(test_loader_auto_no_target_MLP, device)
acc_mode = torch.sum(torch.LongTensor(pred_label_mode) == torch.LongTensor(true_label_mode)) / len(true_label_mode)
print('集成模型(mode)准确率: {:.2f}%'.format(acc_mode * 100)) # 

# 原模型预测准确率
MLP_history = my_test.test(model_MLP, criterion, test_loader_auto_no_target_MLP, device)
DNN_history = my_test.test(model_DNN, criterion, test_loader_auto_no_target_MLP, device)
LSTM_history = my_test.test(model_LSTM, criterion, test_loader_auto_no_target_MLP, device)
print(MLP_history['test']['accuracy'],DNN_history['test']['accuracy'],LSTM_history['test']['accuracy'])


# Autoattack + 无目标 + DNN 生成的对抗样本

# In[ ]:


print('Autoattack + 无目标 + DNN 生成的对抗样本')
pred_label_max, true_label_max = ensemble_max(test_loader_auto_no_target_DNN, device)
acc_max = torch.sum(torch.LongTensor(pred_label_max) == torch.LongTensor(true_label_max)) / len(true_label_max)
print('集成模型(max)准确率: {:.2f}%'.format(acc_max * 100)) # 

pred_label_mean, true_label_mean = ensemble_mean(test_loader_auto_no_target_DNN, device)
acc_mean = torch.sum(torch.LongTensor(pred_label_mean) == torch.LongTensor(true_label_mean)) / len(true_label_mean)
print('集成模型(mean)准确率: {:.2f}%'.format(acc_mean * 100)) # 

pred_label_mode, true_label_mode = ensemble_mode(test_loader_auto_no_target_DNN, device)
acc_mode = torch.sum(torch.LongTensor(pred_label_mode) == torch.LongTensor(true_label_mode)) / len(true_label_mode)
print('集成模型(mode)准确率: {:.2f}%'.format(acc_mode * 100)) # 

# 原模型预测准确率
MLP_history = my_test.test(model_MLP, criterion, test_loader_auto_no_target_DNN, device)
DNN_history = my_test.test(model_DNN, criterion, test_loader_auto_no_target_DNN, device)
LSTM_history = my_test.test(model_LSTM, criterion, test_loader_auto_no_target_DNN, device)
print(MLP_history['test']['accuracy'],DNN_history['test']['accuracy'],LSTM_history['test']['accuracy'])


# Autoattack + 无目标 + LSTM 生成的对抗样本

# In[ ]:


print('Autoattack + 无目标 + LSTM 生成的对抗样本')
pred_label_max, true_label_max = ensemble_max(test_loader_auto_no_target_LSTM, device)
acc_max = torch.sum(torch.LongTensor(pred_label_max) == torch.LongTensor(true_label_max)) / len(true_label_max)
print('集成模型(max)准确率: {:.2f}%'.format(acc_max * 100)) # 

pred_label_mean, true_label_mean = ensemble_mean(test_loader_auto_no_target_LSTM, device)
acc_mean = torch.sum(torch.LongTensor(pred_label_mean) == torch.LongTensor(true_label_mean)) / len(true_label_mean)
print('集成模型(mean)准确率: {:.2f}%'.format(acc_mean * 100)) # 

pred_label_mode, true_label_mode = ensemble_mode(test_loader_auto_no_target_LSTM, device)
acc_mode = torch.sum(torch.LongTensor(pred_label_mode) == torch.LongTensor(true_label_mode)) / len(true_label_mode)
print('集成模型(mode)准确率: {:.2f}%'.format(acc_mode * 100)) # 

# 原模型预测准确率
MLP_history = my_test.test(model_MLP, criterion, test_loader_auto_no_target_LSTM, device)
DNN_history = my_test.test(model_DNN, criterion, test_loader_auto_no_target_LSTM, device)
LSTM_history = my_test.test(model_LSTM, criterion, test_loader_auto_no_target_LSTM, device)
print(MLP_history['test']['accuracy'],DNN_history['test']['accuracy'],LSTM_history['test']['accuracy'])


# Autoattack + 有目标 + MLP 生成的对抗样本

# In[ ]:


print('Autoattack + 有目标 + MLP 生成的对抗样本')
pred_label_max, true_label_max = ensemble_max(test_loader_auto_target_MLP, device)
acc_max = torch.sum(torch.LongTensor(pred_label_max) == torch.LongTensor(true_label_max)) / len(true_label_max)
print('集成模型(max)准确率: {:.2f}%'.format(acc_max * 100)) # 

pred_label_mean, true_label_mean = ensemble_mean(test_loader_auto_target_MLP, device)
acc_mean = torch.sum(torch.LongTensor(pred_label_mean) == torch.LongTensor(true_label_mean)) / len(true_label_mean)
print('集成模型(mean)准确率: {:.2f}%'.format(acc_mean * 100)) # 

pred_label_mode, true_label_mode = ensemble_mode(test_loader_auto_target_MLP, device)
acc_mode = torch.sum(torch.LongTensor(pred_label_mode) == torch.LongTensor(true_label_mode)) / len(true_label_mode)
print('集成模型(mode)准确率: {:.2f}%'.format(acc_mode * 100)) # 

# 原模型预测准确率
MLP_history = my_test.test(model_MLP, criterion, test_loader_auto_target_MLP, device)
DNN_history = my_test.test(model_DNN, criterion, test_loader_auto_target_MLP, device)
LSTM_history = my_test.test(model_LSTM, criterion, test_loader_auto_target_MLP, device)
print(MLP_history['test']['accuracy'],DNN_history['test']['accuracy'],LSTM_history['test']['accuracy'])


# Autoattack + 有目标 + DNN 生成的对抗样本

# In[ ]:


print('Autoattack + 有目标 + DNN 生成的对抗样本')
pred_label_max, true_label_max = ensemble_max(test_loader_auto_target_DNN, device)
acc_max = torch.sum(torch.LongTensor(pred_label_max) == torch.LongTensor(true_label_max)) / len(true_label_max)
print('集成模型(max)准确率: {:.2f}%'.format(acc_max * 100)) # 

pred_label_mean, true_label_mean = ensemble_mean(test_loader_auto_target_DNN, device)
acc_mean = torch.sum(torch.LongTensor(pred_label_mean) == torch.LongTensor(true_label_mean)) / len(true_label_mean)
print('集成模型(mean)准确率: {:.2f}%'.format(acc_mean * 100)) # 

pred_label_mode, true_label_mode = ensemble_mode(test_loader_auto_target_DNN, device)
acc_mode = torch.sum(torch.LongTensor(pred_label_mode) == torch.LongTensor(true_label_mode)) / len(true_label_mode)
print('集成模型(mode)准确率: {:.2f}%'.format(acc_mode * 100)) # 

# 原模型预测准确率
MLP_history = my_test.test(model_MLP, criterion, test_loader_auto_target_DNN, device)
DNN_history = my_test.test(model_DNN, criterion, test_loader_auto_target_DNN, device)
LSTM_history = my_test.test(model_LSTM, criterion, test_loader_auto_target_DNN, device)
print(MLP_history['test']['accuracy'],DNN_history['test']['accuracy'],LSTM_history['test']['accuracy'])


# Autoattack + 有目标 + LSTM 生成的对抗样本

# In[ ]:


print('Autoattack + 有目标 + LSTM 生成的对抗样本')
pred_label_max, true_label_max = ensemble_max(test_loader_auto_target_LSTM, device)
acc_max = torch.sum(torch.LongTensor(pred_label_max) == torch.LongTensor(true_label_max)) / len(true_label_max)
print('集成模型(max)准确率: {:.2f}%'.format(acc_max * 100)) # 

pred_label_mean, true_label_mean = ensemble_mean(test_loader_auto_target_LSTM, device)
acc_mean = torch.sum(torch.LongTensor(pred_label_mean) == torch.LongTensor(true_label_mean)) / len(true_label_mean)
print('集成模型(mean)准确率: {:.2f}%'.format(acc_mean * 100)) # 

pred_label_mode, true_label_mode = ensemble_mode(test_loader_auto_target_LSTM, device)
acc_mode = torch.sum(torch.LongTensor(pred_label_mode) == torch.LongTensor(true_label_mode)) / len(true_label_mode)
print('集成模型(mode)准确率: {:.2f}%'.format(acc_mode * 100)) # 

# 原模型预测准确率
MLP_history = my_test.test(model_MLP, criterion, test_loader_auto_target_LSTM, device)
DNN_history = my_test.test(model_DNN, criterion, test_loader_auto_target_LSTM, device)
LSTM_history = my_test.test(model_LSTM, criterion, test_loader_auto_target_LSTM, device)
print(MLP_history['test']['accuracy'],DNN_history['test']['accuracy'],LSTM_history['test']['accuracy'])


# UAP对抗样本

# UAP0.1

# In[ ]:


print('UAP + 采样test恶意 生成的对抗样本')
pred_label_max, true_label_max = ensemble_max(test_loader_uap_mlp1, device)
acc_max = torch.sum(torch.LongTensor(pred_label_max) == torch.LongTensor(true_label_max)) / len(true_label_max)
print('集成模型(max)准确率: {:.2f}%'.format(acc_max * 100)) # 

pred_label_mean, true_label_mean = ensemble_mean(test_loader_uap_mlp1, device)
acc_mean = torch.sum(torch.LongTensor(pred_label_mean) == torch.LongTensor(true_label_mean)) / len(true_label_mean)
print('集成模型(mean)准确率: {:.2f}%'.format(acc_mean * 100)) # 

pred_label_mode, true_label_mode = ensemble_mode(test_loader_uap_mlp1, device)
acc_mode = torch.sum(torch.LongTensor(pred_label_mode) == torch.LongTensor(true_label_mode)) / len(true_label_mode)
print('集成模型(mode)准确率: {:.2f}%'.format(acc_mode * 100))  # 

MLP_history = my_test.test(model_MLP, criterion, test_loader_uap_mlp1, device)
DNN_history = my_test.test(model_DNN, criterion, test_loader_uap_mlp1, device)
LSTM_history = my_test.test(model_LSTM, criterion, test_loader_uap_mlp1, device)
print(MLP_history['test']['accuracy'],DNN_history['test']['accuracy'],LSTM_history['test']['accuracy'])


# UAP0.2

# In[ ]:


print('UAP + 采样test恶意 生成的对抗样本')
pred_label_max, true_label_max = ensemble_max(test_loader_uap_mlp2, device)
acc_max = torch.sum(torch.LongTensor(pred_label_max) == torch.LongTensor(true_label_max)) / len(true_label_max)
print('集成模型(max)准确率: {:.2f}%'.format(acc_max * 100)) # 

pred_label_mean, true_label_mean = ensemble_mean(test_loader_uap_mlp2, device)
acc_mean = torch.sum(torch.LongTensor(pred_label_mean) == torch.LongTensor(true_label_mean)) / len(true_label_mean)
print('集成模型(mean)准确率: {:.2f}%'.format(acc_mean * 100)) # 

pred_label_mode, true_label_mode = ensemble_mode(test_loader_uap_mlp2, device)
acc_mode = torch.sum(torch.LongTensor(pred_label_mode) == torch.LongTensor(true_label_mode)) / len(true_label_mode)
print('集成模型(mode)准确率: {:.2f}%'.format(acc_mode * 100))  # 

MLP_history = my_test.test(model_MLP, criterion, test_loader_uap_mlp2, device)
DNN_history = my_test.test(model_DNN, criterion, test_loader_uap_mlp2, device)
LSTM_history = my_test.test(model_LSTM, criterion, test_loader_uap_mlp2, device)
print(MLP_history['test']['accuracy'],DNN_history['test']['accuracy'],LSTM_history['test']['accuracy'])


# ### 测试对原test数据集的分类准确率

# In[ ]:


print('原test数据集')
pred_label_max, true_label_max = ensemble_max(test_loader, device)
acc_max = torch.sum(torch.LongTensor(pred_label_max) == torch.LongTensor(true_label_max)) / len(true_label_max)
print('集成模型(max)准确率: {:.2f}%'.format(acc_max * 100)) # 

pred_label_mean, true_label_mean = ensemble_mean(test_loader, device)
acc_mean = torch.sum(torch.LongTensor(pred_label_mean) == torch.LongTensor(true_label_mean)) / len(true_label_mean)
print('集成模型(mean)准确率: {:.2f}%'.format(acc_mean * 100)) # 

pred_label_mode, true_label_mode = ensemble_mode(test_loader, device)
acc_mode = torch.sum(torch.LongTensor(pred_label_mode) == torch.LongTensor(true_label_mode)) / len(true_label_mode)
print('集成模型(mode)准确率: {:.2f}%'.format(acc_mode * 100))  # 

MLP_history = my_test.test(model_MLP, criterion, test_loader, device)
DNN_history = my_test.test(model_DNN, criterion, test_loader, device)
LSTM_history = my_test.test(model_LSTM, criterion, test_loader, device)
print(MLP_history['test']['accuracy'],DNN_history['test']['accuracy'],LSTM_history['test']['accuracy'])


# test数据集原良性样本

# In[ ]:


print('test数据集原良性样本')
pred_label_max, true_label_max = ensemble_max(test_benign_loader, device)
acc_max = torch.sum(torch.LongTensor(pred_label_max) == torch.LongTensor(true_label_max)) / len(true_label_max)
print('集成模型(max)准确率: {:.2f}%'.format(acc_max * 100)) # 

pred_label_mean, true_label_mean = ensemble_mean(test_benign_loader, device)
acc_mean = torch.sum(torch.LongTensor(pred_label_mean) == torch.LongTensor(true_label_mean)) / len(true_label_mean)
print('集成模型(mean)准确率: {:.2f}%'.format(acc_mean * 100)) # 

pred_label_mode, true_label_mode = ensemble_mode(test_benign_loader, device)
acc_mode = torch.sum(torch.LongTensor(pred_label_mode) == torch.LongTensor(true_label_mode)) / len(true_label_mode)
print('集成模型(mode)准确率: {:.2f}%'.format(acc_mode * 100))  # 

MLP_history = my_test.test(model_MLP, criterion, test_benign_loader, device)
DNN_history = my_test.test(model_DNN, criterion, test_benign_loader, device)
LSTM_history = my_test.test(model_LSTM, criterion, test_benign_loader, device)
print(MLP_history['test']['accuracy'],DNN_history['test']['accuracy'],LSTM_history['test']['accuracy'])


# test数据集原恶意样本

# In[ ]:


print('test数据集原恶意样本')
pred_label_max, true_label_max = ensemble_max(test_malicious_loader, device)
acc_max = torch.sum(torch.LongTensor(pred_label_max) == torch.LongTensor(true_label_max)) / len(true_label_max)
print('集成模型(max)准确率: {:.2f}%'.format(acc_max * 100)) # 

pred_label_mean, true_label_mean = ensemble_mean(test_malicious_loader, device)
acc_mean = torch.sum(torch.LongTensor(pred_label_mean) == torch.LongTensor(true_label_mean)) / len(true_label_mean)
print('集成模型(mean)准确率: {:.2f}%'.format(acc_mean * 100)) # 

pred_label_mode, true_label_mode = ensemble_mode(test_malicious_loader, device)
acc_mode = torch.sum(torch.LongTensor(pred_label_mode) == torch.LongTensor(true_label_mode)) / len(true_label_mode)
print('集成模型(mode)准确率: {:.2f}%'.format(acc_mode * 100))  # 

MLP_history = my_test.test(model_MLP, criterion, test_malicious_loader, device)
DNN_history = my_test.test(model_DNN, criterion, test_malicious_loader, device)
LSTM_history = my_test.test(model_LSTM, criterion, test_malicious_loader, device)
print(MLP_history['test']['accuracy'],DNN_history['test']['accuracy'],LSTM_history['test']['accuracy'])

