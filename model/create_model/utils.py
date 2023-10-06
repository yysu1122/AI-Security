import numpy as np
import logging
import os
import json

import torch


def mkdir(dir: str):    
    """建立一个路径为'dir'的目录（如果该目录不存在）"""
    
    if not os.path.exists(dir):
        logging.debug(f"The following path {dir} doesn't exist.")
        os.makedirs(dir)
        logging.debug(f"{dir} successfully created.")


def set_seed(seed: int):
    """ 设置种子 """
     
    np.random.seed(seed)  # random模块的随机数种子
    torch.manual_seed(seed)  # 为CPU设置随机数种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 为GPU设置随机数种子
        torch.cuda.manual_seed_all(seed)  # 为所有的GPU设置随机数种子

def write_json(content, filename):   # 写json文件
    with open(filename, 'w') as write_file:
        json.dump(content, write_file)     


def read_json(filename):            # 读json文件
    with open(filename, 'r') as read_file:
        return json.load(read_file)


