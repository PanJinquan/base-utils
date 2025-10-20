# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : 2025-07-08 14:10:15
# @Brief  : 转换labelme标注数据为voc格式
# --------------------------------------------------------
"""
import os
import numpy as np
from pybaseutils.converter import convert_labelme2voc
from pybaseutils import time_utils

import numpy as np
from collections import deque
from scipy.signal import convolve


def get_valid_range(data, v=0):
    """
    获得值大于v的最大区间
    :param data:
    :param v:
    :return: [index0,index1) 左包含，右不包含
    """
    if not isinstance(data, np.ndarray): data = np.array(data)
    index = np.where(data > v)[0]
    ranges = (0, 0)
    if len(index) > 0:
        ranges = (index[0], index[-1] + 1)
    return ranges


def get_window_sum(score, times, stage_time, ksize=5):
    """
    :param score:
    :param ksize:
    :return:
    """
    score = np.array(score)
    # i1, i2 = get_valid_range(score, v=0)
    ksize = np.ones(ksize)  # 创建滑动窗口核
    # 使用卷积计算滑动窗口和,'same'模式保持输出长度与输入相同
    result = convolve(score, ksize, mode='same')
    # result = result * (score != 0)  # 只有当data[i]不为0时才保留窗口和
    n = len(result)
    area = 0
    time = stage_time
    i = 0
    while i < n:
        if result[i] > 0:
            c0 = i
            while i < n and result[i] > 0:  # 找到事件的开始和结束
                i += 1
            c1 = i
            wsum = sum(result[c0:c1])  # [c0,c1), 左包含，右不包含
            print(wsum, (c0, c1))
            if wsum > area:
                time = (times[c0], times[c1 - 1])
                area = wsum
        i += 1
    print("time={}".format(time))
    return time, result


if __name__ == "__main__":
    score = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 3, 2, 0, 0, 1]
    score = np.asarray(score, dtype=np.float32)
    times = np.arange(len(score))
    stage_time = (0, len(score) - 1)
    ksize = 2
    print("function:", get_window_sum(score, times, stage_time=stage_time, ksize=ksize))
    print("原始数据  :", score)
