import os
import logging
import logging.config
import sys
sys.path.append("..")
from model.create_model import utils


def setup_logging(save_dir, log_config='./loggers/logger_config.json'):
    """ 设置日志记录的配置 """
    config = utils.read_json(log_config)
    for _, handler in config['handlers'].items():
        if 'filename' in handler:
            handler['filename'] = os.path.join(save_dir, handler['filename'])  #在logs文件夹下添加logs.log

    logging.config.dictConfig(config)  #通过字典参数config对logging进行配置
