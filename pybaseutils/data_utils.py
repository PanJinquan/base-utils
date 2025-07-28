# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : 2025-07-22 13:53:44
# @Brief  :
# --------------------------------------------------------
"""

import numpy as np


def is_increase(data):
    """
    判断data是否是递增数列: 每个元素是否大于或等于前一个元素
    :param data:
    :return:
    """
    if len(data) <= 1: return True
    return all(data[i] >= data[i - 1] for i in range(1, len(data)))


def get_orderly_outliers(data: list):
    """
    递增有序序列，查找异常值
    :param data:
    :return:
    """
    nums = len(data)
    r = 1
    indexes = []
    for i in range(1, nums):
        x1 = max(0, i - r)
        x2 = min(nums, i + r + 1)
        window = data[x1:x2]
        if not is_increase(window):
            indexes.append(i)
    return indexes


def get_outliers(data: list, win_size=3, th=3):
    """
    基于滑动窗口+中位数的方法查找异常值
    :param data:
    :param win_size:
    :param th:
    :return:
    """
    nums = len(data)
    r = win_size // 2
    indexes = []
    for i in range(nums):
        x1 = max(0, i - r)
        x2 = min(nums, i + r + 1)
        window = data[x1:x2]
        median = np.median(window)  # 计算窗口中位数
        bias = abs(data[i] - median)  # 计算绝对偏差
        if bias > th:  # 如果偏差过大，则标记为异常
            indexes.append(i)
    return indexes


def median_filter(data, win_size=3):
    """
    一维数据的中值滤波
    也可以直接使用from scipy.ndimage import median_filter
    :param data:
    :param win_size:
    :return:
    """
    nums = len(data)
    r = win_size // 2
    smooth = []
    for i in range(nums):
        x1 = max(0, i - r)
        x2 = min(nums, i + r + 1)
        window = data[x1:x2]
        median = np.median(window)  # 计算中位数
        smooth.append(median)
    return smooth


if __name__ == "__main__":
    from pybaseutils import json_utils

    # data1 = {0: 1739955600.0, 2: 1739937601.0, 3: 1739955602.0}
    # print(data1)
    # data2 = json_utils.dict_sort(data1, use_key=False)
    # print(data2)
    # 示例
    data = [3, 2, 3, 4, 5, 6, 7, 800, 801, 802]
    index = get_orderly_outliers(data)
    print([data[i] for i in index])  # 输出: [3, 7]
