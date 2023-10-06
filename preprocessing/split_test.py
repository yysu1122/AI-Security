# 将test数据集划分为良性流量和恶意流量.恶意流量用于生成对抗样本
# SUYUANYUAN 20230608

import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd

# 加载数据集
DATA_DIR  = os.path.join(os.path.abspath("."), "data")

# 划分
def divide(features, labels):
    benign_features = []
    benign_labels = []
    malicious_features = []
    malicious_labels = []

    for i in range(len(labels)):
        if labels[i] == 0:
            benign_features.append(features[i])
            benign_labels.append(labels[i])
        if labels[i] >= 1 and labels[i] <= 7:  # 由于清洗后数据集保留了八个标签
            malicious_features.append(features[i])
            malicious_labels.append(labels[i])
    

    return benign_features, benign_labels , malicious_features, malicious_labels


if __name__ == '__main__':

    # 加载测试集特征和标签
    test_features = pd.read_pickle(f"{DATA_DIR}/processed/test/test_features.pkl")
    test_labels = pd.read_pickle(f"{DATA_DIR}/processed/test/test_labels.pkl")

    # 划分
    benign_features, benign_labels , malicious_features, malicious_labels = divide(np.array(test_features), np.array(test_labels))

    # 将良性流量和恶意流量分别保存
    test_benign_features = pd.DataFrame(benign_features,columns=test_features.columns)    
    test_benign_labels = pd.DataFrame(benign_labels,columns=test_labels.columns)
    export_benign = pd.concat([test_benign_features,test_benign_labels],axis=1)
    export_benign.to_csv(f"{DATA_DIR}/processed/test/test_benign.csv") # 导入在一个.csv文件里

    test_benign_features.to_pickle(f"{DATA_DIR}/processed/test/test_benign_features.pkl")
    test_benign_labels.to_pickle(f"{DATA_DIR}/processed/test/test_benign_labels.pkl")

    test_malicious_features = pd.DataFrame(malicious_features,columns=test_features.columns)    
    test_malicious_labels = pd.DataFrame(malicious_labels,columns=test_labels.columns)
    export_malicious = pd.concat([test_malicious_features,test_malicious_labels],axis=1)
    export_malicious.to_csv(f"{DATA_DIR}/processed/test/test_malicious.csv")

    test_malicious_features.to_pickle(f"{DATA_DIR}/processed/test/test_malicious_features.pkl")
    test_malicious_labels.to_pickle(f"{DATA_DIR}/processed/test/test_malicious_labels.pkl")

    # print(test_malicious_labels.value_counts())