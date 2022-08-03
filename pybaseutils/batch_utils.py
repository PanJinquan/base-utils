# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-06-06 14:34:38
    @Brief  :
"""
from typing import Tuple, List, Dict


def get_batch_sample(data_list: List, batch_size: int):
    """
    获得的一个batch-size的数据
    batch size data
    :param data_list:
    :param batch_size:
    :return:
    """
    sample_num = len(data_list)
    batch_num = (sample_num + batch_size - 1) // batch_size
    for i in range(batch_num):
        start = i * batch_size
        end = min((i + 1) * batch_size, sample_num)
        batch = data_list[start:end]
        yield batch


def get_batch_dict_sample(data_dict: Dict, batch_size: int):
    """
    获得的一个batch-size的数据
    batch size data
    :param data_dict:
    :param batch_size:
    :return:
    """
    keys = list(data_dict.keys())
    sample_num = len(data_dict[keys[0]])
    batch_num = (sample_num + batch_size - 1) // batch_size
    for i in range(batch_num):
        start = i * batch_size
        end = min((i + 1) * batch_size, sample_num)
        # batch = {k: data_dict[k][start:end] if len(data_dict[k]) > 0 else [] for k in keys}
        batch = {k: data_dict[k][start:end] for k in keys}
        yield batch


def get_batch_example(data_list, batch_size):
    """
    batch size data
    :param data_list:
    :param batch_size:
    :return:
    """
    sample_num = len(data_list)
    batch_num = (sample_num + batch_size - 1) // batch_size
    for i in range(batch_num):
        start = i * batch_size
        end = min((i + 1) * batch_size, sample_num)
        batch = data_list[start:end]
        print("batch:{}".format(batch))


if __name__ == "__main__":
    image_list = ["{}.jpg".format(i) for i in range(10)]
    batch_size = 4
    # get_batch_example(image_list, batch_size)
    for batch in get_batch_sample(image_list, batch_size):
        print("batch:{}".format(batch))

    # data_dict = {"image": image_list, "label": list(range(len(image_list)))}
    data_dict = {"image": image_list, "label": [0, 1, 2]}
    for batch in get_batch_dict_sample(data_dict, batch_size):
        print("batch:{}".format(batch))
