#!/usr/bin/env python
# coding: utf-8

# # 用test数据集的恶意流量数据生成UAP,扰动限制0.2
# # 测试UAP添加到test数据集上的分类准确率变化

# In[ ]:


import torch
import numpy as np
import os.path
import pandas as pd
import sys

sys.path.append("../")
from model.models import MLP,DNN,LSTM
import  gc
gc.collect()
torch.cuda.empty_cache()


# In[ ]:

DATA_DIR  = os.path.join(os.path.abspath("../"), "data")
print(DATA_DIR)

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device ='cpu'
num_classes = 8


# In[ ]:


# 加载模型
model_MLP = MLP(49,64,64,8)
model_MLP.load_state_dict(torch.load('../model/create_model/created_models/MLP.pt'))
model_MLP.to(device)

model_DNN = DNN()
model_DNN.to(device)
model_DNN.load_state_dict(torch.load('../model/create_model/created_models/DNN.pt'))

model_LSTM = LSTM(49,64,8,3)
model_LSTM.to(device)
model_LSTM.load_state_dict(torch.load('../model/create_model/created_models/LSTM.pt'))


# 加载test数据集中的恶意流量

# In[ ]:


from model.create_model import dataset
test_malicious_features_path = f"{DATA_DIR}/processed/test/test_malicious_features.pkl"
test_malicious_labels_path = f"{DATA_DIR}/processed/test/test_malicious_labels.pkl"

test_malicious_data = dataset.CICIDSDataset(
            features_file=test_malicious_features_path,
            target_file=test_malicious_labels_path,
            transform=torch.tensor,
            target_transform=torch.tensor
            )
criterion = torch.nn.CrossEntropyLoss()
batch_size = 64
num_classes = 8
test_malicious_loader = torch.utils.data.DataLoader(dataset=test_malicious_data, batch_size=batch_size, shuffle=False)


# ## UAP攻击方法

# In[ ]:


# deepfool对多分类器的攻击
def deepfool(traffic, model, grads, num_classes=8, overshoot=0.02, max_iter=10):

    """
       :param traffic: traffic of size HxWx3 大小为 HxWx3 的图像
       :param f: feedforward function (input: traffics, output: values of activation BEFORE softmax).
                    前馈功能（输入：图像，输出：softmax之前的激活值）
       :param grads: gradient functions with respect to input (as many gradients as classes).
                    相对于输入的梯度函数（与类一样多的梯度）。
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
                    num_classes（限制要测试的类数，默认情况下 = 10）
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
                    用作终止条件以防止更新消失（默认值 = 0.02）
       :param max_iter: maximum number of iterations for deepfool (default = 10)
                    DeepFool 的最大迭代次数（默认值 = 10）
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed traffic
                    欺骗分类器的最小扰动、所需的迭代次数、新estimated_label和扰动图像
    """

    f_traffic = model(traffic).squeeze().detach().cpu().numpy()  # 预测结果
    I =  f_traffic.argsort()[::-1]         # 对应索引从大到小，大的就是预测label

    I = I[0:num_classes]
    label = I[0]           # 标签1

    input_shape = traffic.shape
    pert_traffic = traffic

    f_i = model(pert_traffic).squeeze().detach().cpu().numpy()
    k_i = int(np.argmax(f_i))         # 标签2

    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    while k_i == label and loop_i < max_iter:

        pert = np.inf

        # 这个梯度是基于预测结果的标签，不是本身正确的标签
        # gradients = np.asarray(grads(pert_traffic,I))    源代码
        traffic_grad = grads(model, pert_traffic, I)             # 利用函数def grad_traffic(model, traffic, targets):
    
        gradients = np.zeros([num_classes,input_shape[1]])
        for k in range(0, num_classes):
            gradients[k] = np.asarray(traffic_grad[k].cpu())

        for k in range(1, num_classes):

            # set new w_k and new f_k
            w_k = gradients[k] - gradients[0]   # 梯度差值
            f_k = f_i[I[k]] - f_i[I[0]]        # 前向反馈后的结果差值，类似于预测置信度差值
            pert_k = abs(f_k)/np.linalg.norm(w_k)  

            # determine which w_k to use  找到可扰动错的最小距离的类别
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot  计算半径ri和总半径r_tot
        r_i =  pert * w / np.linalg.norm(w)
        r_tot = r_tot + r_i

        # compute new perturbed traffic  计算扰动
        pert_traffic = traffic.detach().cpu() + (1+overshoot)*r_tot
        loop_i += 1

        # 预测新标签
        pert_traffic = pert_traffic.clone().detach().requires_grad_(True) 
        pert_traffic = pert_traffic.float().clone().to(device)
        pert_traffic.retain_grad()

        f_i = model(pert_traffic).squeeze().detach().cpu().numpy()
        k_i = int(np.argmax(f_i))

    r_tot = (1+overshoot)*r_tot

    return r_tot, loop_i, k_i, pert_traffic


# In[ ]:


def proj_lp(v, xi, p):
    # Project on the lp ball centered at 0 and of radius xi 
    # SUPPORTS only p = 2 and p = Inf for now  2范数和无穷范数
    if p == 2:
        v = v * min(1, xi/np.linalg.norm(v))
        # v = v / np.linalg.norm(v.flatten(1)) * xi
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
         raise ValueError('Values of p different from 2 and Inf are currently not supported...')

    return v

def universal_perturbation(dataset,  model, grads, delta=0.2, max_iter_uni = np.inf, xi=1, p=np.inf, num_classes=8, overshoot=0.02, max_iter_df=10):
    """
    :param dataset: Images of size MxHxWxC (M: number of images) 大小为 HxWx3 的图像  M：图像数

    :param f: feedforward function (input: images, output: values of activation BEFORE softmax).  前向反馈的值

    :param grads: gradient functions with respect to input (as many gradients as classes). 相对于输入的梯度函数（与类一样多的梯度）

    :param delta: controls the desired fooling rate (default = 80% fooling rate)  控制所需的愚弄率

    :param max_iter_uni: optional other termination criterion (maximum number of iteration, default = np.inf)
                        可选的其他终止条件（最大迭代次数，默认值 = np.inf）

    :param xi: controls the l_p magnitude of the perturbation (default = 10)  控制扰动的l_p幅度（默认值 = 10）

    :param p: norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!) (default = np.inf)  范数 2/inf

    :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
                        （限制要测试的类数，默认情况下 = 10）

    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
                        用作终止条件以防止更新消失（默认值 = 0.02）

    :param max_iter_df: maximum number of iterations for deepfool (default = 10)
                        DeepFool 的最大迭代次数（默认值 = 10）

    :return: the universal perturbation.   返回通用扰动
    """

    v = 0.0
    fooling_rate = 0.0
    num_traffic =  np.shape(dataset)[0] # 样本数
    
    itr = 0
    while fooling_rate < 1-delta and itr < max_iter_uni:
        # Shuffle the dataset 洗牌
        #np.random.shuffle(dataset)
        print ('开始次数 ', itr)

        # 浏览数据集并按顺序计算扰动增量
        for k in range(0, num_traffic):
            cur_traffic = dataset[k:(k+1)]    # 当前第k条流量
            if torch.is_tensor(cur_traffic) == 0:
                cur_traffic = torch.FloatTensor(cur_traffic[cur_traffic.columns.tolist()].to_numpy())
            cur_traffic = cur_traffic.to(device)

            if isinstance(v, float)  and v == 0.0:
                 cur_traffic_v = cur_traffic + v
            else:
                if torch.is_tensor(v) == 0 :
                    v = torch.FloatTensor(v)
                v = v.to(device)
                cur_traffic_v = torch.add(cur_traffic,v)

            # 为了确保类型一致，再计算一遍
            cur_traffic = cur_traffic.to(dtype=torch.float32)
            cur_traffic_v = cur_traffic_v.to(dtype=torch.float32)
            
            # 如果当前预测和添加扰动后的预测一样，需要重新计算到最近的决策边界的距离
            if int(np.argmax(model(cur_traffic).detach().cpu().numpy())) == int(np.argmax(model(cur_traffic_v).detach().cpu().numpy())):
                print('>> 第k条流量, k = ', k, ', pass #', itr)

                # 计算对抗性扰动
                dr,iter,_,_ = deepfool(cur_traffic_v, model, grads, num_classes=num_classes, overshoot=overshoot, max_iter=max_iter_df)

                # 确保它收敛...
                if iter < max_iter_df-1:  # iter没有到最大迭代次数，就已经错误分类了，说明此时扰动可被叠加
                    if isinstance(v, float)  and v == 0.0:
                        v = v + dr
                    else:
                        v = v.cpu().numpy()
                        v = v + dr

                    # Project on l_p 
                    v = proj_lp(v, xi, p)   # 扰动计算p范数后在xi范围内，限制扰动

        itr = itr + 1

        #  使用计算的扰动数据集
        
        if torch.is_tensor(dataset) == 0:
            dataset = torch.FloatTensor(np.array(dataset[dataset.columns.tolist()]))
        dataset_perturbed = torch.add(dataset,v)

        # print('dataset_perturbed\n',dataset_perturbed)

        est_labels_orig = np.zeros((num_traffic))
        est_labels_pert = np.zeros((num_traffic))

        batch_size = 64
        num_batches = int(float(num_traffic) / float(batch_size)) + 1

        #  批量计算估计的标签
        for ii in range(0, num_batches):
            m = (ii * batch_size)
            M = min((ii+1)*batch_size, num_traffic)
            x1 = dataset[m:M]
            x2 = dataset_perturbed[m:M]
            # if torch.cuda.is_available():
            x1, x2 = x1.to(device), x2.to(device)
            est_labels_orig[m:M] = np.argmax(model(x1).detach().cpu(), axis=1)  # 原始预测的标签
            est_labels_pert[m:M] = np.argmax(model(x2).detach().cpu(), axis=1) # 扰动后的预测标签

        # 计算愚弄率
        fooling_rate = float(np.sum(est_labels_pert != est_labels_orig) / float(num_traffic))
        print('愚弄率 = ', fooling_rate)

    return v


# In[ ]:


# 计算预测对某条流量梯度
def grad_traffic(model,  traffic, targets):
    model.to(device)
    traffic.to(device)
    targets = torch.FloatTensor(targets.copy())
    len = (int(np.int64((targets.size()))))
    
    traffic_grad = []  # 不同梯度
    for k in range(0,len):
        i = int(targets[k])
        target = torch.LongTensor([i])
        target = target.to(device)

        traffic.requires_grad_(True)
        traffic.retain_grad()
        outputs = model(traffic)

        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, target)  # 计算损失，输出和不同标签的损失

        model.zero_grad()   # 梯度为0
        loss.backward()  
      
        traffic_grad.extend(traffic.grad.clone().detach())       # 收集梯度,要用clone把梯度单独拿到新的空间里

    return traffic_grad


# ## 生成UAP

# In[ ]:


# UAP和MLP生成对抗样本（参数可设置）
v_MLP = universal_perturbation(test_malicious_data.features, model_MLP, grad_traffic, delta=0.3, xi=0.2,num_classes=num_classes,max_iter_df=30)
# max_iter_uni = 1
print(v_MLP)

# UAP和DNN生成对抗样本
# v_DNN = universal_perturbation(test_malicious_data.features, model_DNN, grad_traffic, delta=0.3, xi=0.2,num_classes=num_classes,max_iter_df=30)

# UAP和LSTM生成对抗样本
# v_LSTM = universal_perturbation(test_malicious_data.features, model_LSTM, grad_traffic, delta=0.3, xi=0.2,num_classes=num_classes,max_iter_df=30)
# print(v_LSTM)


# ## 将UAP添加到测试样本的恶意流量中,保存对抗样本

# In[ ]:


test_malicious_features_tensor= torch.FloatTensor(np.array(test_malicious_data.features))
# 添加UAP_MLP并保存
test_malicious_features_uap_v = torch.add(test_malicious_features_tensor,v_MLP)
test_malicious_features_uap_v = pd.DataFrame((test_malicious_features_uap_v), columns=test_malicious_data.features.columns)
test_malicious_features_uap_v.to_pickle('adversarial_examples/adv_ex_final/test_malicious_UAP0.2_MLP.pkl')
# # 添加UAP_DNN并保存
# test_malicious_features_uap_v = torch.add(test_malicious_features_tensor,v_DNN)
# test_malicious_features_uap_v = pd.DataFrame((test_malicious_features_uap_v), columns=test_malicious_data.features.columns)
# test_malicious_features_uap_v.to_pickle(os.path.join(os.path.abspath(""),'test_malicious_UAP0.2_DNN.pkl'))

# # 添加UAP_LSTM并保存
# test_malicious_features_uap_v = torch.add(test_malicious_features_tensor,v_LSTM)
# test_malicious_features_uap_v = pd.DataFrame((test_malicious_features_uap_v), columns=test_malicious_data.features.columns)
# test_malicious_features_uap_v.to_pickle(os.path.join(os.path.abspath(""),'test_malicious_UAP0.2_LSTM.pkl'))


# ## 测试对抗样本对MLP和DNN，LSTM的攻击效果

# In[ ]:


from model.create_model import my_test,dataset

# UAP+MLP对抗样本
ADSPATH = os.path.join(os.path.abspath(""),'adversarial_examples/adv_ex_final/test_malicious_UAP0.2_MLP.pkl')

test_data_v_MLP = dataset.CICIDSDataset(
            features_file=f"{ADSPATH}",
            target_file=test_malicious_labels_path,
            transform=torch.tensor,
            target_transform=torch.tensor
            )


test_loader = torch.utils.data.DataLoader(dataset=test_data_v_MLP, batch_size=batch_size, shuffle=False)

history = my_test.test(model_MLP, criterion, test_loader, device)
history = my_test.test(model_DNN, criterion, test_loader, device)
history = my_test.test(model_LSTM, criterion, test_loader, device)

# test_output_true = history['test']['output_true']
# test_output_pred = history['test']['output_pred']
# test_output_pred_prob = history['test']['output_pred_prob']
# print(history['test']['accuracy'])

# from sklearn.metrics import classification_report
# labels = ['Benign', 'Botnet ARES', 'Brute Force','DDoS','DoS', 'Infiltration','PortScan','Web Attack']
# print("Testing Set -- Classification Report", end="\n\n")
# print(classification_report(test_output_true, test_output_pred, target_names=labels))


# %%
