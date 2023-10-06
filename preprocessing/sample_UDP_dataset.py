#!/usr/bin/env python
# coding: utf-8

# # 构成防御采样数据集
# # 重新清洗数据集，然后按照类别采样数据。合并类别，构成防御采用数据集
# (有八种流量)
# -------------------------------------------------------------------

# In[ ]:
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append("..")
sys.path.append("../..")
from preprocessing import clean_CICIDS2017


# 加载数据

# In[ ]:
DATA_DIR  = os.path.join(os.path.abspath(".."), "data")
IMAGE_DIR = os.path.join(os.path.abspath(".."), "images")
print(DATA_DIR)
print(IMAGE_DIR)


# In[ ]:
# 清洗数据
cicids2017 = clean_CICIDS2017.CICIDS2017Preprocessor(
    data_path=DATA_DIR,
    training_size=0.6,
    validation_size=0.2,
    testing_size=0.2
)

# 读数据
cicids2017.read_data()

# 移除重复值，确实值，无穷值
cicids2017.remove_duplicate_values()
cicids2017.remove_missing_values
cicids2017.remove_infinite_values()

# 删除常数列及高度相关的特征列
cicids2017.remove_constant_features()
cicids2017.remove_correlated_features()


# In[ ]:

cicids2017.data['label'] = cicids2017.data['label'].str.replace('Web Attack �', 'Web Attack', regex=False)
#分开特征和标签
X_data = pd.DataFrame(cicids2017.data.drop(labels=['label'], axis=1)) #pd.DataFrame(dataset, columns=columns) 
y_data = pd.DataFrame(cicids2017.data['label'] )                  # pd.DataFrame(dataset, columns=["label"]) 


# 正常流量偏多，采样一定比例  
# 样本较多的恶意流量，随机采样部分  
# 样本较少的恶意流量，全部采样使用之后，再合并到别的相似流量类型中（eg:DoS Hulk -> DoS, Web Attack XSS -> Web Attack）

# In[ ]:
# 采样策略
undersampling_strategy = {
    'BENIGN': 20000,
    'DoS Hulk':1000,
    'PortScan':3000,
    'DDoS':4000,
    'DoS GoldenEye':1000,
    'FTP-Patator':1500,
    'SSH-Patator':1500,
    'DoS slowloris':1000,
    'DoS Slowhttptest':1000,
    'Bot':1500,
    'Web Attack Brute Force':1000,
    'Web Attack XSS':500,
    'Infiltration':36,
    'Web Attack Sql Injection':21,
    'Heartbleed':11,
}


# In[ ]:

from imblearn.under_sampling import RandomUnderSampler
under_sampler = RandomUnderSampler(sampling_strategy=undersampling_strategy,random_state=0)
X_under, y_under = under_sampler.fit_resample(X_data, y_data)


# 对采样后的数据合并少数类，便于后期利用已有模型
# In[ ]:
attack_group = {
    'BENIGN': 'Benign',
    'PortScan': 'PortScan',
    'DDoS': 'DDoS',
    'DoS Hulk': 'DoS',
    'DoS GoldenEye': 'DoS',
    'DoS slowloris': 'DoS', 
    'DoS Slowhttptest': 'DoS',
    'Heartbleed': 'DoS',
    'FTP-Patator': 'Brute Force',
    'SSH-Patator': 'Brute Force',
    'Bot': 'Botnet ARES',
    'Web Attack Brute Force': 'Web Attack',
    'Web Attack Sql Injection': 'Web Attack',
    'Web Attack XSS': 'Web Attack',
    'Infiltration': 'Infiltration'
}

# 建立组标签列
y_under['label'] = y_under['label'].map(lambda x: attack_group[x])
y_under['label'].value_counts()


# In[ ]:
# 拟合+归一化

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, QuantileTransformer
from sklearn.compose import ColumnTransformer

categorical_features = X_under.select_dtypes(exclude=["int64", "float64"]).columns
numeric_features = X_under.select_dtypes(exclude=[object]).columns

preprocessor = ColumnTransformer(transformers=[
    ('categoricals', OneHotEncoder(drop='first', sparse=False, handle_unknown='error'), categorical_features),
    ('numericals', QuantileTransformer(), numeric_features)
])

columns = numeric_features.tolist()
X = pd.DataFrame(preprocessor.fit_transform(X_under), columns=columns)

le = LabelEncoder()
y = pd.DataFrame(le.fit_transform(y_under), columns=["label"])


# In[ ]:

y.value_counts()

# In[ ]:

# 保存采样后的数据
X.to_pickle(os.path.join(DATA_DIR, 'processed','defense_data/under_sampler_feature.pkl'))
y.to_pickle(os.path.join(DATA_DIR, 'processed','defense_data/under_sampler_label.pkl'))

