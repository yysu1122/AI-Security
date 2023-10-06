#!/usr/bin/env python
# coding: utf-8

# 所用模型：DNN  

# # 用前期采样的流量数据生成UDP（部分流量，涵盖所有类别）

# In[ ]:


import torch
import torch.nn as nn
import numpy as np

import os
import sys
sys.path.append("..")
sys.path.append("../..")
from model.models import MLP,DNN,LSTM
from model.create_model import dataset,visualisation


# In[ ]:


DATA_DIR  = os.path.join(os.path.abspath(".."), "data")
IMAGE_DIR = os.path.join(os.path.abspath(""), "udp_images")
traffic_feature_path = os.path.join(DATA_DIR, 'processed','defense_data/under_sampler_feature.pkl')
traffic_label_path = os.path.join(DATA_DIR, 'processed','defense_data/under_sampler_label.pkl')

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device ='cpu'
num_classes = 8


# In[ ]:


# 加载模型
# model = MLP(49,64,64,8)
# model.load_state_dict(torch.load('../model/create_model/created_models/MLP.pt'))

model = DNN()
model.load_state_dict(torch.load('../model/create_model/created_models/DNN.pt'))

# model = LSTM()
# model.load_state_dict(torch.load('../model/create_model/created_models/LSTM.pt'))
model.to(device)


# In[ ]:


train_data, val_data, test_data = dataset.get_dataset(data_path=DATA_DIR, balanced=True)
criterion = torch.nn.CrossEntropyLoss()
batch_size = 64

# 加载数据
from model.create_model import dataset
traffic_data = dataset.CICIDSDataset(
            features_file=f"{traffic_feature_path}",
            target_file=f"{traffic_label_path}",
            transform=torch.tensor,
            target_transform=torch.tensor
            )
# 加载流量样本
traffic_loader = torch.utils.data.DataLoader(dataset=traffic_data, batch_size=batch_size, shuffle=False)
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


# In[ ]:


# 生成一个叠加的统一的UDP
def udpd(benign_samples, labels, classifier, eps, max_epochs):
    udp = torch.zeros_like(benign_samples[0])  # 初始化通用逆扰动,49维
    udp_epoch = []  # 用作观察数据
    a = eps/max_epochs #阿尔法
   
    n_samples = benign_samples.shape[0]  # 样本数量
    #num_classes = classifier(benign_samples).shape[1]  # 类型数量

    for epoch in range(max_epochs):
        for i in range(n_samples):
            x_i = benign_samples[i]
            x_i_udp = torch.add(x_i, udp)

            x_i_udp.requires_grad = True

            label = torch.LongTensor([int(labels[i])])
            label = label.squeeze()
            # 计算交叉熵损失
            output = classifier(x_i_udp)
            loss = torch.nn.functional.cross_entropy(output, label) #.squeeze()
            
            # 计算梯度
            grad = torch.autograd.grad(loss, x_i_udp)[0]  # 公式(8)

            # 计算通用逆扰动
            # grad = grad / torch.norm(grad, p=2)  
            sign_data_grad = grad.sign()  # 收集数据梯度的逐元素符号

            # 更新通用逆扰动
            udp -= a*sign_data_grad
        
        # print(udp)
        # udp = torch.clamp(udp, -0.1, 0.1)
        udp_epoch.append(udp)

    return udp_epoch


# In[ ]:


classifier = model
eps = 0.0005
max_epochs = 30
traffic_data_features = torch.FloatTensor(np.array(traffic_data.features[traffic_data.features.columns.tolist()]))
traffic_data_labels = torch.FloatTensor(np.array(traffic_data.labels[traffic_data.labels.columns.tolist()]))


# In[ ]:


# 使用采样数据集生成UDP
rho_udp = udpd(traffic_data_features, traffic_data_labels, classifier, eps, max_epochs)
threshold = 0.000001

rho_udp = rho_udp[-1]
print(rho_udp)


# ## 将UDP添加到测试样本中，测试预测准确率

# 测试集添加UDP，观察是否影响正常流量的检测（正常良性流量和正常恶意流量）  
# 目标：不影响正常流量检测性能

# In[ ]:


import pandas as pd
from model.create_model import my_test,dataset,utils
from sklearn.metrics import classification_report


# 原模型对原test数据集的分类情况

# In[ ]:


test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
history = my_test.test(model, criterion, test_loader, device)

test_output_true = history['test']['output_true']
test_output_pred = history['test']['output_pred']
test_output_pred_prob = history['test']['output_pred_prob']
print(history['test']['accuracy'])

labels = ['Benign', 'Botnet ARES', 'Brute Force','DDoS','DoS', 'Infiltration','PortScan','Web Attack']
print("Classification Report", end="\n\n")
print(classification_report(test_output_true, test_output_pred, target_names=labels))

visualisation.plot_confusion_matrix(y_true=test_output_true,
                                    y_pred=test_output_pred,
                                    labels=labels,
                                    save=True,
                                    save_dir=IMAGE_DIR,
                                    filename='test_confusion_matrix.svg')


# In[ ]:


# 加载测试集，测试集添加UDP，保存
batch_size = 64
test_data_features_tensor = torch.FloatTensor(np.array(test_data.features[test_data.features.columns.tolist()]))
test_data_features_udp = torch.add(test_data_features_tensor,rho_udp)

test_data_features_udp = pd.DataFrame((test_data_features_udp), columns=test_data.features.columns)
test_data_features_udp.to_pickle(os.path.join(os.path.abspath(""),'test_data_features_udp.pkl'))

test_ex_udp_temp_PATH = os.path.join(os.path.abspath(""),'test_data_features_udp.pkl')

test_ex_udp_temp = dataset.CICIDSDataset(
            features_file=f"{test_ex_udp_temp_PATH}",
            target_file=f"{DATA_DIR}/processed/test/test_labels.pkl",
            transform=torch.tensor,
            target_transform=torch.tensor
            )


# 原模型对 test+UDP 的分类情况 

# In[ ]:


test_loader = torch.utils.data.DataLoader(dataset=test_ex_udp_temp, batch_size=batch_size, shuffle=False)
history = my_test.test(model, criterion, test_loader, device)

test_output_true = history['test']['output_true']
test_output_pred = history['test']['output_pred']
test_output_pred_prob = history['test']['output_pred_prob']
print(history['test']['accuracy'])

labels = ['Benign', 'Botnet ARES', 'Brute Force','DDoS','DoS', 'Infiltration','PortScan','Web Attack']
print("Classification Report", end="\n\n")
print(classification_report(test_output_true, test_output_pred, target_names=labels))

visualisation.plot_confusion_matrix(y_true=test_output_true,
                                    y_pred=test_output_pred,
                                    labels=labels,
                                    save=True,
                                    save_dir=IMAGE_DIR,
                                    filename='test_udp_confusion_matrix.svg')


# 对原良性流量样本的分类情况

# In[ ]:


history = my_test.test(model, criterion, test_benign_loader, device)

test_output_true = history['test']['output_true']
test_output_pred = history['test']['output_pred']
test_output_pred_prob = history['test']['output_pred_prob']
print(history['test']['accuracy'])
labels = ['Benign', 'Botnet ARES', 'Brute Force','DDoS','DoS', 'Infiltration','PortScan','Web Attack']
print("Classification Report", end="\n\n")
print(classification_report(test_output_true, test_output_pred, target_names=labels))


# 良性流量添加UDP后的分类情况

# In[ ]:


test_benign_features_tensor= torch.FloatTensor(np.array(test_benign_data.features))

# 添加UDP并保存
test_benign_features_udp = torch.add(test_benign_features_tensor,rho_udp)
test_benign_features_udp = pd.DataFrame((test_benign_features_udp), columns=test_benign_data.features.columns)
test_benign_features_udp.to_pickle(os.path.join(os.path.abspath(""),'test_benign_features_udp.pkl'))

test_benign_udp_temp_PATH = os.path.join(os.path.abspath(""),'test_benign_features_udp.pkl')

test_benign_udp_temp = dataset.CICIDSDataset(
            features_file=f"{test_benign_udp_temp_PATH}",
            target_file=f"{DATA_DIR}/processed/test/test_benign_labels.pkl",
            transform=torch.tensor,
            target_transform=torch.tensor
            )


# In[ ]:


test_loader_temp = torch.utils.data.DataLoader(dataset=test_benign_udp_temp, batch_size=batch_size, shuffle=False)
history = my_test.test(model, criterion, test_loader_temp, device)

test_output_true = history['test']['output_true']
test_output_pred = history['test']['output_pred']
test_output_pred_prob = history['test']['output_pred_prob']
print(history['test']['accuracy'])

labels = ['Benign', 'Botnet ARES', 'Brute Force','DDoS','DoS', 'Infiltration','PortScan','Web Attack']
print("Classification Report", end="\n\n")
print(classification_report(test_output_true, test_output_pred, target_names=labels))


# 对原恶意流量样本的分类情况

# In[ ]:


history = my_test.test(model, criterion, test_malicious_loader, device)

test_output_true = history['test']['output_true']
test_output_pred = history['test']['output_pred']
test_output_pred_prob = history['test']['output_pred_prob']
print(history['test']['accuracy'])
labels = ['Benign', 'Botnet ARES', 'Brute Force','DDoS','DoS', 'Infiltration','PortScan','Web Attack']
print("Classification Report", end="\n\n")
print(classification_report(test_output_true, test_output_pred, target_names=labels))


# 恶意流量添加UDP后的分类情况

# In[ ]:


test_malicious_features_tensor= torch.FloatTensor(np.array(test_malicious_data.features))

# 添加UAP并保存
test_malicious_features_udp = torch.add(test_malicious_features_tensor,rho_udp)
test_malicious_features_udp = pd.DataFrame((test_malicious_features_udp), columns=test_malicious_data.features.columns)
test_malicious_features_udp.to_pickle(os.path.join(os.path.abspath(""),'test_malicious_features_udp.pkl'))

test_malicious_udp_temp_PATH = os.path.join(os.path.abspath(""),'test_malicious_features_udp.pkl')

test_malicious_udp_temp = dataset.CICIDSDataset(
            features_file=f"{test_malicious_udp_temp_PATH}",
            target_file=f"{DATA_DIR}/processed/test/test_malicious_labels.pkl",
            transform=torch.tensor,
            target_transform=torch.tensor
            )


# In[ ]:


test_loader_temp = torch.utils.data.DataLoader(dataset=test_malicious_udp_temp, batch_size=batch_size, shuffle=False)
history = my_test.test(model, criterion, test_loader_temp, device)

test_output_true = history['test']['output_true']
test_output_pred = history['test']['output_pred']
test_output_pred_prob = history['test']['output_pred_prob']
print(history['test']['accuracy'])

labels = ['Benign', 'Botnet ARES', 'Brute Force','DDoS','DoS', 'Infiltration','PortScan','Web Attack']
print("Classification Report", end="\n\n")
print(classification_report(test_output_true, test_output_pred, target_names=labels))


# ## 将UDP添加到FGSM对抗样本中，测试预测准确率

# In[ ]:


# 加载test数据集中的恶意流量和标签
test_malicious_features_path = f"{DATA_DIR}/processed/test/test_malicious_features.pkl"
test_malicious_labels_path = f"{DATA_DIR}/processed/test/test_malicious_labels.pkl"


# In[ ]:


# 读取对抗样本集数据
adv_ex_features_PATH = os.path.join(os.path.abspath("../adversarial_attacks/adversarial_examples/adv_ex_final"),'adv_ex_features_FGSM_MLP.pkl')
# adv_ex_features_PATH = os.path.join(os.path.abspath("../adversarial_attacks/adversarial_examples/adv_ex_final"),'adv_ex_features_FGSM_DNN.pkl')
# adv_ex_features_PATH = os.path.join(os.path.abspath("../adversarial_attacks/adversarial_examples/adv_ex_final"),'adv_ex_features_FGSM_LSTM.pkl')
adv_ex_features = pd.read_pickle(adv_ex_features_PATH)
adv_ex_features = torch.FloatTensor(np.array(adv_ex_features))

# 添加udp并保存
adv_ex_udp_temp_features = torch.add(adv_ex_features,rho_udp)
adv_ex_udp_temp_features = pd.DataFrame((adv_ex_udp_temp_features), columns=traffic_data.features.columns)
adv_ex_udp_temp_features.to_pickle(os.path.join(os.path.abspath(""),'adv_ex_udp_temp_features.pkl'))

# 设置数据集
adv_ex_udp_temp_PATH = os.path.join(os.path.abspath(""),'adv_ex_udp_temp_features.pkl')
adv_ex_udp_temp = dataset.CICIDSDataset(
            features_file=f"{adv_ex_udp_temp_PATH}",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,
            target_transform=torch.tensor
            )
# 设置迭代加载器
test_loader_temp = torch.utils.data.DataLoader(dataset=adv_ex_udp_temp, batch_size=batch_size, shuffle=False)

# 预测分类
history = my_test.test(model, criterion, test_loader_temp, device)

test_output_true = history['test']['output_true']
test_output_pred = history['test']['output_pred']
test_output_pred_prob = history['test']['output_pred_prob']
print(history['test']['accuracy'])

print("Classification Report", end="\n\n")
print(classification_report(test_output_true, test_output_pred, target_names=labels))


# ## 将UDP添加到NI-FGSM对抗样本中，测试预测准确率

# In[ ]:


# 读取对抗样本集数据
adv_ex_features_PATH = os.path.join(os.path.abspath("../adversarial_attacks/adversarial_examples/adv_ex_final"),'adv_ex_features_NI_FGSM_MLP.pkl')
# adv_ex_features_PATH = os.path.join(os.path.abspath("../adversarial_attacks/adversarial_examples/adv_ex_final"),'adv_ex_features_NI_FGSM_DNN.pkl')
# adv_ex_features_PATH = os.path.join(os.path.abspath("../adversarial_attacks/adversarial_examples/adv_ex_final"),'adv_ex_features_NI_FGSM_LSTM.pkl')
adv_ex_features = pd.read_pickle(adv_ex_features_PATH)
adv_ex_features = torch.FloatTensor(np.array(adv_ex_features))

# 添加udp并保存
adv_ex_udp_temp_features = torch.add(adv_ex_features,rho_udp)
adv_ex_udp_temp_features = pd.DataFrame((adv_ex_udp_temp_features), columns=traffic_data.features.columns)
adv_ex_udp_temp_features.to_pickle(os.path.join(os.path.abspath(""),'adv_ex_udp_temp_features.pkl'))

# 设置数据集
adv_ex_udp_temp_PATH = os.path.join(os.path.abspath(""),'adv_ex_udp_temp_features.pkl')
adv_ex_udp_temp = dataset.CICIDSDataset(
            features_file=f"{adv_ex_udp_temp_PATH}",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,
            target_transform=torch.tensor
            )
# 设置迭代加载器
test_loader_temp = torch.utils.data.DataLoader(dataset=adv_ex_udp_temp, batch_size=batch_size, shuffle=False)

# 预测分类
history = my_test.test(model, criterion, test_loader_temp, device)

test_output_true = history['test']['output_true']
test_output_pred = history['test']['output_pred']
test_output_pred_prob = history['test']['output_pred_prob']
print(history['test']['accuracy'])

print("Classification Report", end="\n\n")
print(classification_report(test_output_true, test_output_pred, target_names=labels))


# ## 将UDP添加到SINI-FGSM对抗样本中，测试预测准确率

# In[ ]:


# 读取对抗样本集数据
adv_ex_features_PATH = os.path.join(os.path.abspath("../adversarial_attacks/adversarial_examples/adv_ex_final"),'adv_ex_features_SI_NI_FGSM_MLP.pkl')
# adv_ex_features_PATH = os.path.join(os.path.abspath("../adversarial_attacks/adversarial_examples/adv_ex_final"),'adv_ex_features_SI_NI_FGSM_DNN.pkl')
# adv_ex_features_PATH = os.path.join(os.path.abspath("../adversarial_attacks/adversarial_examples/adv_ex_final"),'adv_ex_features_SI_NI_FGSM_LSTM.pkl')
adv_ex_features = pd.read_pickle(adv_ex_features_PATH)
adv_ex_features = torch.FloatTensor(np.array(adv_ex_features))

# 添加UDP并保存
adv_ex_udp_temp_features = torch.add(adv_ex_features,rho_udp)
adv_ex_udp_temp_features = pd.DataFrame((adv_ex_udp_temp_features), columns=traffic_data.features.columns)
adv_ex_udp_temp_features.to_pickle(os.path.join(os.path.abspath(""),'adv_ex_udp_temp_features.pkl'))

# 设置数据集
adv_ex_udp_temp_PATH = os.path.join(os.path.abspath(""),'adv_ex_udp_temp_features.pkl')
adv_ex_udp_temp = dataset.CICIDSDataset(
            features_file=f"{adv_ex_udp_temp_PATH}",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,
            target_transform=torch.tensor
            )
# 设置迭代加载器
test_loader_temp = torch.utils.data.DataLoader(dataset=adv_ex_udp_temp, batch_size=batch_size, shuffle=False)

history = my_test.test(model, criterion, test_loader_temp, device)

test_output_true = history['test']['output_true']
test_output_pred = history['test']['output_pred']
test_output_pred_prob = history['test']['output_pred_prob']
print(history['test']['accuracy'])

print("Classification Report", end="\n\n")
print(classification_report(test_output_true, test_output_pred, target_names=labels))


# In[ ]:


# 测试对原来对抗样本的分类情况
adv_ex = dataset.CICIDSDataset(
            features_file=f"{adv_ex_features_PATH}",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,
            target_transform=torch.tensor
            )
test_loader_temp = torch.utils.data.DataLoader(dataset=adv_ex, batch_size=batch_size, shuffle=False)
history = my_test.test(model, criterion, test_loader_temp, device)

test_output_true = history['test']['output_true']
test_output_pred = history['test']['output_pred']
test_output_pred_prob = history['test']['output_pred_prob']
print(history['test']['accuracy'])

print(classification_report(test_output_true, test_output_pred, target_names=labels))


# ------------
# ## 将UDP添加到Autoattack对抗样本中，测试预测准确率

# 
# ### 添加到基于 no-target + MLP/DNN/LSTM的样本

# In[ ]:


# 读取对抗样本集数据
adv_ex_features_PATH = os.path.join(os.path.abspath("../adversarial_attacks/adversarial_examples/adv_ex_final"),'adv_ex_features_auto_no_target_MLP.pkl')
# adv_ex_features_PATH = os.path.join(os.path.abspath("../adversarial_attacks/adversarial_examples/adv_ex_final"),'adv_ex_features_auto_no_target_DNN.pkl')
# adv_ex_features_PATH = os.path.join(os.path.abspath("../adversarial_attacks/adversarial_examples/adv_ex_final"),'adv_ex_features_auto_no_target_LSTM.pkl')

adv_ex_features = pd.read_pickle(adv_ex_features_PATH)
adv_ex_features = torch.FloatTensor(np.array(adv_ex_features))

# 添加udp并保存
adv_ex_udp_temp_features = torch.add(adv_ex_features,rho_udp)
adv_ex_udp_temp_features = pd.DataFrame((adv_ex_udp_temp_features), columns=traffic_data.features.columns)
adv_ex_udp_temp_features.to_pickle(os.path.join(os.path.abspath(""),'adv_ex_udp_temp_features.pkl'))


# In[ ]:


# 设置数据集
adv_ex_udp_temp_PATH = os.path.join(os.path.abspath(""),'adv_ex_udp_temp_features.pkl')
adv_ex_udp_temp = dataset.CICIDSDataset(
            features_file=f"{adv_ex_udp_temp_PATH}",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,
            target_transform=torch.tensor
            )
# 设置迭代加载器
test_loader_temp = torch.utils.data.DataLoader(dataset=adv_ex_udp_temp, batch_size=batch_size, shuffle=False)

history = my_test.test(model, criterion, test_loader_temp, device)

test_output_true = history['test']['output_true']
test_output_pred = history['test']['output_pred']
test_output_pred_prob = history['test']['output_pred_prob']
print(history['test']['accuracy'])

print("Classification Report", end="\n\n")
print(classification_report(test_output_true, test_output_pred, target_names=labels))


# In[ ]:


# 测试对原来对抗样本的分类情况
adv_ex = dataset.CICIDSDataset(
            features_file=f"{adv_ex_features_PATH}",
            target_file=test_malicious_labels_path,   # 基于test恶意流量的正确的标签
            transform=torch.tensor,
            target_transform=torch.tensor
            )
test_loader_temp = torch.utils.data.DataLoader(dataset=adv_ex, batch_size=batch_size, shuffle=False)
history = my_test.test(model, criterion, test_loader_temp, device)

test_output_true = history['test']['output_true']
test_output_pred = history['test']['output_pred']
test_output_pred_prob = history['test']['output_pred_prob']
print(history['test']['accuracy'])

print(classification_report(test_output_true, test_output_pred, target_names=labels))


# 
# ### 添加到基于 target + MLP/DNN/LSTM的样本

# In[ ]:


# 读取对抗样本集数据
adv_ex_features_PATH = os.path.join(os.path.abspath("../adversarial_attacks/adversarial_examples/adv_ex_final"),'adv_ex_features_auto_target_MLP.pkl')
# adv_ex_features_PATH = os.path.join(os.path.abspath("../adversarial_attacks/adversarial_examples/adv_ex_final"),'adv_ex_features_auto_target_DNN.pkl')
# adv_ex_features_PATH = os.path.join(os.path.abspath("../adversarial_attacks/adversarial_examples/adv_ex_final"),'adv_ex_features_auto_target_LSTM.pkl')

adv_ex_features = pd.read_pickle(adv_ex_features_PATH)
adv_ex_features = torch.FloatTensor(np.array(adv_ex_features))

# 添加udp并保存
adv_ex_udp_temp_features = torch.add(adv_ex_features,rho_udp)
adv_ex_udp_temp_features = pd.DataFrame((adv_ex_udp_temp_features), columns=traffic_data.features.columns)
adv_ex_udp_temp_features.to_pickle(os.path.join(os.path.abspath(""),'adv_ex_udp_temp_features.pkl'))


# In[ ]:


# 设置数据集
adv_ex_udp_temp_PATH = os.path.join(os.path.abspath(""),'adv_ex_udp_temp_features.pkl')
adv_ex_udp_temp = dataset.CICIDSDataset(
            features_file=f"{adv_ex_udp_temp_PATH}",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,
            target_transform=torch.tensor
            )
# 设置迭代加载器
test_loader_temp = torch.utils.data.DataLoader(dataset=adv_ex_udp_temp, batch_size=batch_size, shuffle=False)

history = my_test.test(model, criterion, test_loader_temp, device)

test_output_true = history['test']['output_true']
test_output_pred = history['test']['output_pred']
test_output_pred_prob = history['test']['output_pred_prob']
print(history['test']['accuracy'])

print("Classification Report", end="\n\n")
print(classification_report(test_output_true, test_output_pred, target_names=labels))


# In[ ]:


# 测试对原来对抗样本的分类情况
adv_ex = dataset.CICIDSDataset(
            features_file=f"{adv_ex_features_PATH}",
            target_file=test_malicious_labels_path,   # 基于test恶意流量的正确的标签
            transform=torch.tensor,
            target_transform=torch.tensor
            )
test_loader_temp = torch.utils.data.DataLoader(dataset=adv_ex, batch_size=batch_size, shuffle=False)
history = my_test.test(model, criterion, test_loader_temp, device)

test_output_true = history['test']['output_true']
test_output_pred = history['test']['output_pred']
test_output_pred_prob = history['test']['output_pred_prob']
print(history['test']['accuracy'])

print(classification_report(test_output_true, test_output_pred, target_names=labels))


# ------------
# ## 将UDP添加到UAP对抗样本中，测试预测准确率

# ### 添加到基于 UAP_0.1/0.2 + MLP 的对抗样本

# In[ ]:


# 读取对抗样本集数据
adv_ex_features_PATH = os.path.join(os.path.abspath("../adversarial_attacks/adversarial_examples/adv_ex_final"),'test_malicious_UAP0.1_MLP')
# adv_ex_features_PATH = os.path.join(os.path.abspath("../adversarial_attacks/adversarial_examples/adv_ex_final"),'test_malicious_UAP0.2_MLP')
adv_ex_features = pd.read_pickle(adv_ex_features_PATH)
adv_ex_features = torch.FloatTensor(np.array(adv_ex_features))

# 添加UDP并保存
adv_ex_udp_temp_features = torch.add(adv_ex_features,rho_udp)
adv_ex_udp_temp_features = pd.DataFrame((adv_ex_udp_temp_features), columns=traffic_data.features.columns)
adv_ex_udp_temp_features.to_pickle(os.path.join(os.path.abspath(""),'adv_ex_udp_temp_features.pkl'))

# 设置数据集
adv_ex_udp_temp_PATH = os.path.join(os.path.abspath(""),'adv_ex_udp_temp_features.pkl')
adv_ex_udp_temp = dataset.CICIDSDataset(
            features_file=f"{adv_ex_udp_temp_PATH}",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,
            target_transform=torch.tensor
            )
# 设置迭代加载器
test_loader_temp = torch.utils.data.DataLoader(dataset=adv_ex_udp_temp, batch_size=batch_size, shuffle=False)

# 预测分类
history = my_test.test(model, criterion, test_loader_temp, device)

test_output_true = history['test']['output_true']
test_output_pred = history['test']['output_pred']
test_output_pred_prob = history['test']['output_pred_prob']
print(history['test']['accuracy'])

print("Classification Report", end="\n\n")
print(classification_report(test_output_true, test_output_pred, target_names=labels))


# In[ ]:


# 测试对原来对抗样本的分类情况
adv_ex = dataset.CICIDSDataset(
            features_file=f"{adv_ex_features_PATH}",
            target_file=test_malicious_labels_path,   # 基于test恶意流量的正确的标签
            transform=torch.tensor,
            target_transform=torch.tensor
            )
test_loader_temp = torch.utils.data.DataLoader(dataset=adv_ex, batch_size=batch_size, shuffle=False)
history = my_test.test(model, criterion, test_loader_temp, device)

test_output_true = history['test']['output_true']
test_output_pred = history['test']['output_pred']
test_output_pred_prob = history['test']['output_pred_prob']
print(history['test']['accuracy'])

print(classification_report(test_output_true, test_output_pred, target_names=labels))


# ## 求UDP的均值和方差

# In[ ]:


print(rho_udp)
# 均值
print(torch.mean(rho_udp))
# 方差
print(torch.var(rho_udp))

