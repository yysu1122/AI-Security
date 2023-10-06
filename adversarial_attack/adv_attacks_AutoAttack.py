#!/usr/bin/env python
# coding: utf-8

# # 使用Autoattack攻击，生成对抗样本

# In[ ]:


import os
import argparse
from pathlib import Path
import warnings
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import sys
sys.path.append('../')
sys.path.append('../..')
from model.models import MLP,DNN,LSTM
from model.create_model import dataset
from tqdm import tqdm
import pandas as pd


# 设置参数

# In[ ]:


DATA_DIR  = os.path.join(os.path.abspath("../"), "data")
print(DATA_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--norm', type=str, default='Linf')  
parser.add_argument('--epsilon', type=float, default=0.1)
parser.add_argument('--model', type=str, default='./MLP.pt')
parser.add_argument('--n_ex', type=int, default=1000)  #设置了1000个检查点
parser.add_argument('--individual', action='store_true')
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--batch_size', type=int, default=1024)  # 每批运行1024个样本
parser.add_argument('--log_path', type=str, default='./log_file.txt')
parser.add_argument('--version', type=str, default='custom')  # standard/custom/rand/plus
parser.add_argument('--state-path', type=Path, default=None)
args = parser.parse_args(args=[])


# 加载test恶意流量，用于生成对抗样本

# In[ ]:


test_malicious_features_path = f"{DATA_DIR}/processed/test/test_malicious_features.pkl"
test_malicious_labels_path = f"{DATA_DIR}/processed/test/test_malicious_labels.pkl"

test_malicious_data = dataset.CICIDSDataset(
            features_file=test_malicious_features_path,
            target_file=test_malicious_labels_path,
            transform=torch.tensor,
            target_transform=torch.tensor
            )

x_test = test_malicious_data.features
y_test = test_malicious_data.labels
x_test_tensor = torch.FloatTensor(x_test.values)
y_test_tensor = torch.LongTensor(y_test.values)


# 加载从训练集采样的样本，用于生成加强对抗样本

# In[ ]:


under_train_features_path = f"{DATA_DIR}/processed/train/under_sampler_train_features_balanced.pkl"
under_train_labels_path = f"{DATA_DIR}/processed/train/under_sampler_train_labels_balanced.pkl"

under_train_data = dataset.CICIDSDataset(
            features_file=under_train_features_path,
            target_file=under_train_labels_path,
            transform=torch.tensor,
            target_transform=torch.tensor
            )

# x_test = under_train_data.features
# y_test = under_train_data.labels
# x_test_tensor = torch.FloatTensor(x_test.values)
# y_test_tensor = torch.LongTensor(y_test.values)


# 加载模型

# In[ ]:


# model = MLP(49, 64, 64, 8)
# model.load_state_dict(torch.load('../model/create_model/created_models/MLP.pt'))

model = DNN()
model.load_state_dict(torch.load('../model/create_model/created_models/DNN.pt'))

# model = LSTM(49,64,8,3)
# model.load_state_dict(torch.load('../model/create_model/created_models/LSTM.pt'))
model.to('cuda')
model.train()

# 建立保存目录
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
    
# 加载攻击    
from autoattack.autoattack import AutoAttack
adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon, log_path=args.log_path,
    version=args.version)

# 自定义版本的测试(把五种攻击分成无目标和有目标测试)
if args.version == 'custom':
    adversary.attacks_to_run = ['apgd-ce', 'apgd-dlr', 'fab'] # 无目标
    # adversary.attacks_to_run = ['apgd-t', 'fab-t'] # 有目标
    # adversary.apgd.n_restarts = 2
    # adversary.fab.n_restarts = 2


# 运行攻击并保存结果

# In[ ]:


device='cuda'
adv_examples = []
args.individual = False
columns = x_test.columns

with torch.no_grad():
    torch.cuda.empty_cache()   # 清除显存
    inputs, labels = x_test_tensor.to(device), y_test_tensor.to(device)
    labels = labels.squeeze(1)
    if not args.individual:

        adv_complete = adversary.run_standard_evaluation(inputs,labels,
            bs=args.batch_size, state_path=args.state_path)
        adv_ex = adv_complete.squeeze().detach().cpu().numpy()

    else:
        # individual version, each attack is run on all test points
        adv_complete = adversary.run_standard_evaluation_individual(inputs,
            labels, bs=args.batch_size)  
        adv_ex = adv_complete.squeeze().detach().cpu().numpy()
        
# 保存对抗样本
adv_ex = pd.DataFrame(adv_ex,columns=columns)
adv_ex.to_pickle('adversarial_examples/adv_ex_final/under_train_adv_ex_auto_no_target_DNN.pkl')


