# -*- coding:utf-8 -*-
"""
提供工具函数的模块
"""

import logging
import yaml

log = logging.getLogger(__name__)


class Dict2Obj:
    '''
    dict转类对象
    '''

    def __init__(self, args):
        self.__dict__.update(args)


def load_config(config_file='config.yaml'):
    """
    读取配置文件，并返回一个python dict 对象
    :param config_file: 配置文件路径
    :return: python dict 对象
    """
    with open(config_file, 'r', encoding="UTF-8") as stream:
        try:
            config = yaml.load(stream, Loader=yaml.FullLoader)
            # config = Dict2Obj(config)
        except yaml.YAMLError as e:
            print(e)
            return None
    return config


def print_dict(dict_data, save_path=None):
    list_config = []
    print("--" * 30)
    for key in dict_data:
        info = "{}: {}".format(key, dict_data[key])
        print(info)
        list_config.append(info)
    if save_path is not None:
        with open(save_path, "w") as f:
            for info in list_config:
                f.writelines(info + "\n")
    print("--" * 30)


if __name__ == '__main__':
    pass
