import pandas as pd
import numpy as np
import glob
import os
import torch

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

#data数据路径
DATA_DIR  = os.path.join(os.path.abspath("."), "data")


class CICIDS2017Preprocessor(object):
    def __init__(self, data_path, training_size, validation_size, testing_size):
        self.data_path = data_path
        self.training_size = training_size
        self.validation_size = validation_size
        self.testing_size = testing_size
        
        self.data = None
        self.features = None
        self.label = None

    def read_data(self):
        """"""
        filenames = glob.glob(os.path.join(self.data_path, 'raw', '*.csv'))  #读出目录下的所有csv文件 ,raw为原始文件目录
        datasets = [pd.read_csv(filename) for filename in filenames]

        # 移除空白，重命名列
        for dataset in datasets:
            dataset.columns = [self._clean_column_name(column) for column in dataset.columns]

        # 连接数据集
        self.data = pd.concat(datasets, axis=0, ignore_index=True)

    def _clean_column_name(self, column):     #列标题处理
   
        column = column.strip(' ') # 去除首尾空格
        column = column.replace('/', '_')  #/变为_
        column = column.replace(' ', '_')
        column = column.lower()     
        return column

    def remove_duplicate_values(self):     # 删除重复行
       
        self.data.drop_duplicates(inplace=True, keep=False, ignore_index=True)   #keep:删除所有重复项，修改数据，从0开始标记

    def remove_missing_values(self):
        
        self.data.dropna(axis=0, inplace=True, how="any")   #删除包含缺失值的行

    def remove_infinite_values(self):
        
        # 将无限值替换为 NaN
        self.data.replace([-np.inf, np.inf], np.nan, inplace=True)  
        # 删除包含无限值的行
        self.data.dropna(axis=0, how='any', inplace=True)   

    def remove_constant_features(self, threshold=0.01):    #移除常量特征  
       
        # Standard deviation denoted by sigma (σ) is the average of the squared root differences from the mean.
        data_std = self.data.std(numeric_only=True)

        # 寻找满足阈值的特征         
        constant_features = [column for column, std in data_std.iteritems() if std < threshold]

        # 丢弃这些常量特征
        self.data.drop(labels=constant_features, axis=1, inplace=True)

    def remove_correlated_features(self, threshold=0.98):  #移除相关元素？
        
        # 相关矩阵
        data_corr = self.data.corr()     #计算列的成对相关性，不包括 NA/null 值。

        # 创建和使用mask矩阵
        mask = np.triu(np.ones_like(data_corr, dtype=bool))     #返回一个bool类型的相同大小的矩阵，mask为对应的上三角矩阵
        tri_df = data_corr.mask(mask)    #应用被罩矩阵

        # 寻找满足阈值的特征
        correlated_features = [c for c in tri_df.columns if any(tri_df[c] > threshold)]

        # 移除高相关性的元素
        self.data.drop(labels=correlated_features, axis=1, inplace=True)

    def group_labels(self):
        """"""
        # 合并少数类
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
            'Web Attack � Brute Force': 'Web Attack',   
            'Web Attack � Sql Injection': 'Web Attack',
            'Web Attack � XSS': 'Web Attack',
            'Infiltration': 'Infiltration'
        }

        # 建立新的标签列
        self.data['label_category'] = self.data['label'].map(lambda x: attack_group[x])
        
    def train_valid_test_split(self):   #划分训练，测试，验证集
        
        self.labels = self.data['label_category']
        self.features = self.data.drop(labels=['label', 'label_category'], axis=1)  

        X_train, X_test, y_train, y_test = train_test_split(
            self.features,
            self.labels,
            test_size=(self.validation_size + self.testing_size),
            random_state=42,                              
            stratify=self.labels
        )
        X_test, X_val, y_test, y_val = train_test_split(
            X_test,
            y_test,
            test_size=self.testing_size / (self.validation_size + self.testing_size),
            random_state=42
        )
    
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def scale(self, training_set, validation_set, testing_set):
        # 划分数据特征的类型，预处理特征和标签

        (X_train, y_train), (X_val, y_val), (X_test, y_test) = training_set, validation_set, testing_set
        
        categorical_features = self.features.select_dtypes(exclude=["number"]).columns  # 类别型特征列
        numeric_features = self.features.select_dtypes(exclude=[object]).columns      # 数值型特征列

        preprocessor = ColumnTransformer(transformers=[
            ('categoricals', OneHotEncoder(drop='first', sparse=False, handle_unknown='error'), categorical_features),    #
            ('numericals', QuantileTransformer(), numeric_features)
        ])

        #drop='first' 删除每个特征中的第一个元素， sparse=False，返回一个数组  handle_unknown='error'如果在转换过程中存在未知类别，则引发错误。
        #QuantileTransformer() 使数据均匀分布


        # 预处理特征
        columns = numeric_features.tolist()

        X_train = pd.DataFrame(preprocessor.fit_transform(X_train), columns=columns)
        X_val = pd.DataFrame(preprocessor.transform(X_val), columns=columns)
        X_test = pd.DataFrame(preprocessor.transform(X_test), columns=columns)

        # 预处理标签
        le = LabelEncoder()

        y_train = pd.DataFrame(le.fit_transform(y_train), columns=["label"])
        y_val = pd.DataFrame(le.transform(y_val), columns=["label"])
        y_test = pd.DataFrame(le.transform(y_test), columns=["label"])

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == "__main__":

    cicids2017 = CICIDS2017Preprocessor(
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

    # 建立新的标签策略
    cicids2017.group_labels()

    # 划分数据集，归一化
    training_set, validation_set, testing_set  = cicids2017.train_valid_test_split()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = cicids2017.scale(training_set, validation_set, testing_set)
    
    # 保存
    X_train.to_pickle(os.path.join(DATA_DIR, 'processed', 'train/train_features.pkl'))
    X_val.to_pickle(os.path.join(DATA_DIR, 'processed', 'val/val_features.pkl'))
    X_test.to_pickle(os.path.join(DATA_DIR, 'processed', 'test/test_features.pkl'))

    y_train.to_pickle(os.path.join(DATA_DIR, 'processed', 'train/train_labels.pkl'))
    y_val.to_pickle(os.path.join(DATA_DIR, 'processed', 'val/val_labels.pkl'))
    y_test.to_pickle(os.path.join(DATA_DIR, 'processed', 'test/test_labels.pkl'))