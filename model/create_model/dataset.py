import pandas as pd
import torch
from torch.utils.data.dataset import Dataset

#from utils import utils


class CICIDSDataset(Dataset):

    def __init__(self, features_file, target_file, transform=None, target_transform=None):
        """
        Args:
            features_file (string): Path to the csv file with features.
            target_file (string): Path to the csv file with labels.
            transform (callable, optional): Optional transform to be applied on features.
            target_transform (callable, optional): Optional transform to be applied on labels.
        """
        self.features = pd.read_pickle(features_file)
        self.labels = pd.read_pickle(target_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):     #取某一行特征及标签
        feature = self.features.iloc[idx, :]  #对数据进行位置索引，从而在数据表中提取出相应的数据[行索引，列索引]
        label = self.labels.iloc[idx]
        if self.transform:
            feature = self.transform(feature.values, dtype=torch.float32)
        if self.target_transform:
            label = self.target_transform(label, dtype=torch.int64)
        return feature, label


def get_dataset(data_path: str, balanced: bool):

    if balanced:
        train_data = CICIDSDataset(
            features_file=f"{data_path}/processed/train/train_features_balanced.pkl",
            target_file=f"{data_path}/processed/train/train_labels_balanced.pkl",
            transform=torch.tensor,
            target_transform=torch.tensor
        )
    else:
        train_data = CICIDSDataset(
            features_file=f"{data_path}/processed/train/train_features.pkl",
            target_file=f"{data_path}/processed/train/train_labels.pkl",
            transform=torch.tensor,
            target_transform=torch.tensor
        )

    val_data = CICIDSDataset(
        features_file=f"{data_path}/processed/val/val_features.pkl",
        target_file=f"{data_path}/processed/val/val_labels.pkl",
        transform=torch.tensor,
        target_transform=torch.tensor
    )

    test_data = CICIDSDataset(
        features_file=f"{data_path}/processed/test/test_features.pkl",
        target_file=f"{data_path}/processed/test/test_labels.pkl",
        transform=torch.tensor,
        target_transform=torch.tensor
    )

    return train_data, val_data, test_data


def load_data(data_path: str, balanced: bool, batch_size: int):    #创建数据加载器，加载训练、验证和测试集

    # 获得数据集
    train_data, val_data, test_data = get_dataset(data_path=data_path, balanced=balanced)

    # 建立加载器
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True
        #num_workers=2  多线程读取数据
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, valid_loader, test_loader
