# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2025-07-08 14:10:15
# @Brief  :
# --------------------------------------------------------
"""
import cv2
import numpy as np
from pybaseutils import file_utils, image_utils

import cv2


def find_optimal_splits(data):
    """
    找到最优分割点i和j，使得 |sum(data2)-sum(data1)| + |sum(data2)-sum(data3)| 最大化

    参数:
        data: 输入列表

    返回:
        (max_value, i, j, data1, data2, data3): 最大值和对应的分割点及切片
    """
    n = len(data)
    max_value = float('-inf')
    best_i, best_j = 0, 0

    # 计算前缀和以便快速计算区间和
    prefix_sum = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix_sum[i] = prefix_sum[i - 1] + data[i - 1]

    # 遍历所有可能的分割点
    for i in range(1, n - 1):  # i从1到n-2，确保data1和data2至少有一个元素
        for j in range(i + 1, n):  # j从i+1到n-1，确保data2和data3至少有一个元素
            sum1 = prefix_sum[i] - prefix_sum[0]  # data[0:i]的和
            sum2 = prefix_sum[j] - prefix_sum[i]  # data[i:j]的和
            sum3 = prefix_sum[n] - prefix_sum[j]  # data[j:]的和

            # 计算目标函数值
            value = abs(sum2 - sum1) + abs(sum2 - sum3)

            if value > max_value:
                max_value = value
                best_i, best_j = i, j

    # 获取最优分割后的数据切片
    data1 = data[:best_i]
    data2 = data[best_i:best_j]
    data3 = data[best_j:]

    return max_value, best_i, best_j, data1, data2, data3


# 测试函数
data = [1, 2, 3, 4, 100, 101, 102, 5, 6, 7, 8]
max_value, i, j, data1, data2, data3 = find_optimal_splits(data)

print(f"最优分割点: i={i}, j={j}")
print(f"最大值: {max_value}")
print(f"data1: {data1}, sum={sum(data1)}")
print(f"data2: {data2}, sum={sum(data2)}")
print(f"data3: {data3}, sum={sum(data3)}")
print(f"验证: |{sum(data2)}-{sum(data1)}| + |{sum(data2)}-{sum(data3)}| = {abs(sum(data2) - sum(data1)) + abs(sum(data2) - sum(data3))}")