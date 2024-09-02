# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2024-08-24 11:05:52
    @Brief  :
"""
import os
import numpy as np
import itertools
from pybaseutils import file_utils, image_utils


def get_combinations_pair_data(image_dir, pair_num=0):
    """
    获得图片对数据
    :param image_dir:
    :param pair_num:
    :return:
    """
    max_nums = int(pair_num / 2)
    image_list = file_utils.get_files_lists(image_dir)
    image_list = file_utils.get_sub_list(image_list, dirname=image_dir)
    nums = len(image_list)
    print("have {} images and {} combinations".format(nums, nums * (nums - 1) / 2))
    pairs = []
    for paths in itertools.combinations(image_list, 2):
        file1, file2 = paths
        label1 = file1.split(os.sep)[0]
        label2 = file2.split(os.sep)[0]
        if label1 == label2:
            pairs.append([file1, file2, 1])
        else:
            pairs.append([file1, file2, 0])
    pairs = np.asarray(pairs)
    pairs = pairs[np.lexsort(pairs.T)]
    pair0 = pairs[pairs[:, -1] == "0", :]
    pair1 = pairs[pairs[:, -1] == "1", :]
    nums1 = len(pair1)
    nums0 = len(pair0)
    if pair_num < 0: max_nums = nums1
    if max_nums > nums1:
        raise Exception("pair_nums({}) must be less than num_pair_issame_1({})".format(max_nums, nums1))
    index_0 = np.random.permutation(nums0)[:max_nums]  # 打乱后的行号
    index_1 = np.random.permutation(nums1)[:max_nums]  # 打乱后的行号
    pair0 = pair0[index_0, :]  # 获取打乱后的训练数据
    pair1 = pair1[index_1, :]  # 获取打乱后的训练数据
    pairs = np.concatenate([pair0, pair1], axis=0).tolist()
    print("have {} pairs，pair0 nums:{}，pair1 nums:{}".format(len(pairs), len(pair0), len(pair1)))
    return pairs


if __name__ == '__main__':
    image_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-action-cvlm/test-v2/images"
    pairs_file = os.path.join(image_dir, "pairs.txt")
    pairs = get_combinations_pair_data(image_dir, pair_num=10000)
    file_utils.write_data(pairs_file, pairs)
