# 基于通用扰动的智能网络入侵检测系统对抗鲁棒性增强技术
课题面向于智能网络入侵检测系统（AI-NIDS）开展对抗样本攻击的防御技术研究，提出了一种应对多类型对抗样本攻击的通用防御方案。利用原始样本数据集和已训练的DNN模型迭代反馈生成用于重识别防御的通用防御扰动（Universal Defense Perturbation，UDP）。将其添加至流量特征样本中，可在几乎不影响正常流量检测的情况下，抵消对抗样本中的攻击扰动，从数据层面消除样本对抗性，有效提高模型对各类别对抗攻击样本的检测准确率。相较于模型受到攻击时的表现，UDP防御后的平均检测准确率可提高35.17%。

## 实验环境
python 版本：3.9.15  
pytorch 版本：1.13.1  
cuda 版本：11.6  

## CICIDS-2017
使用[CICIDS-2017数据集](https://www.unb.ca/cic/datasets/ids-2017.html)的原始特征文件（.csv）建立干净的数据集。其中包含良性流量数据特征（Benign）以及最常见的攻击流量特征，共有2,830,743条记录，分布在8个文件中，每条记录包含78种不同的特征及其标签。 

**数据集预处理** *"preprocessing"*
+ **清洗数据** *"clean_CICIDS2017.py"*：删除数据集中的重复样本，具有缺失值和无穷值的样本。删除常数列和高度相关的特征列，避免带来多余的信息。清洗后的样本共2,425,727条，特征列保留了49项。合并数据集中具有相似特征的少数类，避免分布不均衡。最终的标签种类数为8种。将处理后的数据集划分为训练集（60%），验证集（20%），测试集（20%）。
+ **平衡训练集** *"balance_train.py"*：使用随机欠采样（RandomUnderSampler）和过采样方法（SMOTE）的组合平衡训练数据集，适当增加训练集中少样本的数量。
+ **降维可视化** *"reduce_feature.py"*：利用PCA降维技术将合并类别后的干净数据集特征可视化到二维空间（Benign类别样本可视化10000个）。
+ **划分测试数据集** *"split_test.py"*：将test数据集划分为良性流量和恶意流量。恶意流量用于生成对抗样本。
+ **采样训练数据集** *"sample_train.py"*：下采样训练集，用采样的数据生成新的对抗样本。利用新对抗样本加强训练集，便于对抗训练。
+ **构建防御采样数据集** *"sample_UDP_dataset.py"*：按照类别设定下采样原始特征数据集，构成UDP防御采样数据集。

## 目标AI-NIDS模型（MLP、DNN、LSTM）
分别搭建多层感知机（MLP）、深度神经网络（DNN）、长短期记忆递归神经网络（LSTM）3 个智能网络入侵检测模型。 *"model"* 

**模型** ：MLP、DNN、LSTM *"models"*     

**训练模型** *"create_model"*

## 对抗样本攻击
将测试集划分为良性流量和恶意流量，利用恶意流量数据生成多种对抗扰动，制作有目标攻击和无目标攻击的对抗样本。攻击方法均在pytorch框架下实现。*"adversarial_attack"*  

+ **FGSM**, "[Explaining and harnessing adversarial examples](https://arxiv.org/abs/1412.6572)"
+ **NI-FGSM**, "[Nesterov Acceralated Gradient and Scale Invariance for Adversarial Attacks (ICLR2020)](https://openreview.net/forum?id=SJlHwkBYDH)"
+ **SINI-FGSM** ,"[Nesterov Acceralated Gradient and Scale Invariance for Adversarial Attacks (ICLR2020)](https://openreview.net/forum?id=SJlHwkBYDH)"
+ **AutoAttack**,"[Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks](https://arxiv.org/abs/2003.01690)",课题取消了 Square 攻击。
+ **UAP**,"[Universal adversarial perturbations](http://arxiv.org/pdf/1610.08401)"

## 通用防御扰动（UDP）
利用防御采样数据集和前期训练好的 DNN模型，迭代提取正常流量样本的全局特征。计算分类器对每次更新后的正常样本的分类损失和梯度，按照一定步长迭代减少梯度以生成 UDP。 *"universal_defense"*  
参考论文：[Adversarial examples are not bugs, they are features](https://arxiv.org/abs/1905.02175)，[基于通用逆扰动的对抗攻击防御方法](http://www.aas.net.cn/article/doi/10.16383/j.aas.c201077)

## 防御方案对比

+ **集成模型** *"ensemble"*：将训练好的模型（MLP，DNN，LSTM）进行集成，利用投票机制得到它们的综合预测结果。分别按照三种模型的outputs的最大值（Max），均值（Mean），和输出标签的众数（Mode）判断样本最终类别。
+ **对抗训练** *"adversarial_train"*：按照类别从原始训练集中采样一部分样本，利用三个模型和两种攻击生成新的对抗样本以增强训练集。假设防御者已知SINI-FGSM型和AutoAttack型攻击，分别将以上两种攻击样本集合添加至原始训练集，对抗训练MLP、DNN、LSTM模型。测试对抗模型的检测效果。
