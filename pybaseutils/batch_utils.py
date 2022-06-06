# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-06-06 14:34:38
    @Brief  :
"""


def get_batch_sample(data_list, batch_size):
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
    batch_size = 15
    get_batch_example(image_list, batch_size)
    for batch in get_batch_sample(image_list, batch_size):
        print("batch:{}".format(batch))
