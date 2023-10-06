#!/usr/bin/env python
# coding: utf-8

# # 特征降维PCA(应该可视化原始特征)
# ------------------------------------------------

# ## 加载数据集

# In[ ]:


import pandas as pd
import numpy as np
import os

# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


DATA_DIR  = os.path.join(os.path.abspath(".."), "data")
IMAGE_DIR = os.path.join(os.path.abspath(".."), "images")
IMAGE_DIR


# In[ ]:


X_train = pd.read_pickle(os.path.join(DATA_DIR, 'processed', 'train/train_features_balanced.pkl'))
X_val = pd.read_pickle(os.path.join(DATA_DIR, 'processed', 'val/val_features.pkl'))
X_test = pd.read_pickle(os.path.join(DATA_DIR, 'processed', 'test/test_features.pkl'))

y_train = pd.read_pickle(os.path.join(DATA_DIR, 'processed', 'train/train_labels_balanced.pkl'))
y_val = pd.read_pickle(os.path.join(DATA_DIR, 'processed', 'val/val_labels.pkl'))
y_test = pd.read_pickle(os.path.join(DATA_DIR, 'processed', 'test/test_labels.pkl'))

data_features = pd.read_pickle(os.path.join(DATA_DIR, 'processed', 'raw/data_features.pkl'))
data_labels = pd.read_pickle(os.path.join(DATA_DIR, 'processed', 'raw/data_labels.pkl'))


# ## PCA 
# 
# 在降维之后，通常不会为每个主成分分配特定的含义。新Component只是变化后的两个主要维度。

# In[ ]:
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# In[ ]:

pca = PCA(n_components=2)

# principalComponents = pca.fit_transform(X_train)
# 数据标准化
# data_features = (data_features - data_features.mean()) / data_features.std()
principalComponents = pca.fit_transform(data_features)
principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

# ### 降维到二维空间以可视化
# 

# In[ ]:

# finalDf = pd.concat([principalDf, y_train], axis=1)
finalDf = pd.concat([principalDf, data_labels], axis=1)

# 绘制非"Benign"类别的散点图
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
# targets = y_train['label'].unique()
targets = ['Benign', 'DoS', 'DDoS', 'PortScan', 'Brute Force', 'Web Attack', 'Botnet ARES', 'Infiltration']#y_test['label'].unique()

for target in [4, 3, 6, 2, 7, 1, 5]:  #[0, 4, 3, 6, 2, 7, 1, 5]
    filt = finalDf['label'] == target  #_category
    ax.scatter(finalDf.loc[filt, 'principal component 1'], finalDf.loc[filt, 'principal component 2'],s=50)

# 绘制"Benign"类别的散点图
benign_filt = finalDf['label'] == 0
benign_indices = finalDf.loc[benign_filt].head(10000).index  # 10000个良性
ax.scatter(finalDf.loc[filt, 'principal component 1'], finalDf.loc[filt, 'principal component 2'],s=50)


# ax.set_xlim(-5, 5)
# ax.set_ylim(-3, 3)
ax.set_xlabel('Principal Component 1', fontsize=10)
ax.set_ylabel('Principal Component 2', fontsize=10)
ax.set_title('2 Component PCA', fontsize=10)
ax.legend(targets)
ax.grid()
fig.savefig(os.path.join(IMAGE_DIR, 'feature_reduction.svg'))


# 降维到三维空间可视化

# In[ ]:


pca3 = PCA(n_components=3)

principalComponents = pca3.fit_transform(data_features)
principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2','principal component 3'])

finalDf = pd.concat([principalDf, data_labels], axis=1)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')


labels = ['Benign', 'DoS', 'DDoS', 'PortScan', 'Brute Force', 'Web Attack', 'Botnet ARES', 'Infiltration']
targets = [ 4, 3, 6, 2, 7, 1, 5]

# colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'orange'] #设置颜色顺序
# for target, color in zip(targets, colors):
#     filt = finalDf['label'] == target
#     ax.scatter(finalDf.loc[filt, 'principal component 1'], finalDf.loc[filt, 'principal component 2'], finalDf.loc[filt, 'principal component 3'], c=color, s=50)

# 绘制非"Benign"类别的散点图
for target in targets:
    filt = finalDf['label'] == target
    ax.scatter(finalDf.loc[filt, 'principal component 1'], finalDf.loc[filt, 'principal component 2'], finalDf.loc[filt, 'principal component 3'], s=50)
# 绘制"Benign"类别的散点图
benign_filt = finalDf['label'] == 0
benign_indices = finalDf.loc[benign_filt].head(100000).index
ax.scatter(finalDf.loc[filt, 'principal component 1'], finalDf.loc[filt, 'principal component 2'],finalDf.loc[filt, 'principal component 3'], s=50)

# ax.set_xlim(-5, 5)
# ax.set_ylim(-3, 3)
# ax.set_zlim(-3, 3)
ax.set_xlabel('Principal Component 1', fontsize=16)
ax.set_ylabel('Principal Component 2', fontsize=16)
ax.set_zlabel('Principal Component 3', fontsize=16)
ax.set_title('3 Component PCA', fontsize=18)
ax.legend(labels)
ax.grid()
# ax.view_init(elev=0, azim=180) # 调整视角。观察到的Benign总是包裹了其他流量，几乎看不到恶意流量（0，90时可以看到Dos）
plt.show()
fig.savefig(os.path.join(IMAGE_DIR, 'feature_reduction3.svg'))


# In[ ]:


print(principalDf.describe())


# ### 解释方差
# 解释的方差看出每个主成分可以获得多少信息（方差）。

# In[ ]:


print(pca.explained_variance_ratio_)
pca3.explained_variance_ratio_


# In[ ]:


print(sum(pca.explained_variance_ratio_))
sum(pca3.explained_variance_ratio_)


#   
# 两个主要成分加在一起，包含了大约63%的信息。第一个主成分包含大约47%的方差。第二个主要成分包含大约16%的方差。

# ### 累积解释方差与主成分数量的关系
# 

# In[ ]:


import seaborn as sns

sns.set_theme(style="white", color_codes=True)


# In[ ]:


pca = PCA()

pca.fit(X_test)

tot = sum(pca.explained_variance_)

var_exp = [(i/tot)*100 for i in sorted(pca.explained_variance_, reverse=True)] 

cum_var_exp = np.cumsum(var_exp)

fig, ax = plt.subplots(figsize=(12, 7))
plt.plot(range(1, 50), cum_var_exp)
#plt.title('Explained Variance by Components', fontsize=18)
plt.ylabel('Cumulative explained variance', fontsize = 18)
plt.xlabel('Principal components', fontsize = 18)
ax.axhline(y=99, color='c', linestyle='--', label='99% explained variance')
ax.axhline(y=97, color='k', linestyle='--', label='97% explained variance')
ax.axhline(y=95, color='r', linestyle='--', label='95% explained variance')
ax.legend(loc='best', markerscale=1.0, fontsize=14)
ax.grid()
# fig.savefig(os.path.join(IMAGE_DIR, 'pca.pdf'))


# ### 保留99%方差的最小主成分数量
# 

# In[ ]:


pca = PCA(0.99)
# numeric_features = X_train.select_dtypes(exclude=[object]).columns

X_train_pca = pd.DataFrame(pca.fit_transform(X_train))
# X_train_pca = pd.DataFrame(pca.fit_transform(X_train),columns=X_train.columns.tolist())
X_val_pca = pd.DataFrame(pca.transform(X_val))
X_test_pca = pd.DataFrame(pca.transform(X_test))


# In[ ]:


sum(pca.explained_variance_ratio_)


# In[ ]:


X_train_pca


# In[ ]:


pca.n_features_in_


# In[ ]:


pca.explained_variance_


# In[ ]:


len(pca.explained_variance_)


# 
# 25个主成分保留了99%的方差，而不是49个。

# ## t-SNE   太慢了

# In[ ]:


# import pandas as pd
# import numpy as np
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

# # # 从csv文件读取数据
# # data = pd.read_csv('CICIDS2017_features.csv')

# # # 分离标签列
# # labels = data['Label']
# # features = data.drop('Label', axis=1)

# # # 标准化特征
# # features = (features - features.mean()) / features.std()

# # 实例化t-SNE并指定目标维度
# tsne = TSNE(n_components=3)

# # 使用fit_transform方法降维
# X_tsne = tsne.fit_transform(X_test)
# # targets = ['Benign', 'DoS', 'DDoS', 'PortScan', 'Brute Force', 'Web Attack', 'Botnet ARES', 'Infiltration']

# # 可视化降维后的数据
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_test, cmap='tab10')
# plt.show()

