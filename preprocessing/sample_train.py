# 下采样训练集，用采样的数据生成新的对抗样本
# 然后重新放入训练集，加强训练集。便于之后对抗训练
# SUYUANYUAN 20230608

import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

# 加载数据集
DATA_DIR  = os.path.join(os.path.abspath("."), "data")

# 加载训练集特征和标签
train_features = pd.read_pickle(f"{DATA_DIR}/processed/train/train_features_balanced.pkl")
train_labels = pd.read_pickle(f"{DATA_DIR}/processed/train/train_labels_balanced.pkl")

# 下采样设置
undersampling_strategy = {
    0: 10000,
    4: 10000,
    3: 10000,
    6: 10000,
    2: 5000,
    7: 5000,
    1: 5000,
    5: 300,
}

under_sampler = RandomUnderSampler(sampling_strategy=undersampling_strategy,random_state=0)
X_under, y_under = under_sampler.fit_resample(train_features, train_labels)

print(y_under.value_counts())

# 保存
X_under.to_pickle(f"{DATA_DIR}/processed/train/under_sampler_train_features_balanced.pkl")
y_under.to_pickle(f"{DATA_DIR}/processed/train/under_sampler_train_labels_balanced.pkl")