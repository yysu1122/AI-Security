from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import os
import pandas as pd

# 数据集路径
DATA_DIR  = os.path.join(os.path.abspath("."), "data")

# 平衡数据集
def balance_dataset(X, y, undersampling_strategy, oversampling_strategy):

    under_sampler = RandomUnderSampler(sampling_strategy=undersampling_strategy, random_state=0)  #抽取数据
    X_under, y_under = under_sampler.fit_resample(X, y)
    #sampling_strategy包含要对数据集进行采样的信息的字典。键对应于要从中采样的类标签，值是要采样的样本数。random_state随机种子
    #X是采样的矩阵，y是每个样本的相应标签

    over_sampler = SMOTE(sampling_strategy=oversampling_strategy)
    X_bal, y_bal = over_sampler.fit_resample(X_under, y_under)    #对下采样数据进行过采样，增加数据
    
    return X_bal, y_bal


if __name__ == '__main__':

    # 下采样设置
    undersampling_strategy = {
        0: 600000,
        4: 115358,
        3: 76803,
        6: 34383,
        2: 5130,
        7: 1271,
        1: 1166,
        5: 22,
    }

    # 过采样
    oversampling_strategy = {
        0: 600000,
        4: 115358,
        3: 76803,
        6: 34383,
        2: 25130,
        7: 21271,
        1: 21166,
        5: 522,
    }

    # 加载测试集特征和标签
    X_train = pd.read_pickle(f"{DATA_DIR}/processed/train/train_features.pkl")
    y_train = pd.read_pickle(f"{DATA_DIR}/processed/train/train_labels.pkl")

    # 平衡训练集
    X_train_bal, y_train_bal = balance_dataset(X_train, y_train, undersampling_strategy, oversampling_strategy)

    # 保存已平衡的训练集
    X_train_bal.to_pickle(os.path.join(DATA_DIR, 'processed', 'train/train_features_balanced.pkl'))
    y_train_bal.to_pickle(os.path.join(DATA_DIR, 'processed', 'train/train_labels_balanced.pkl'))