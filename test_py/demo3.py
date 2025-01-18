# -*- coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-08-30 09:45:44
    @Brief  :
"""

import os
import numpy as np
import cv2
from tqdm import tqdm
from pybaseutils import image_utils, file_utils, time_utils
from pybaseutils.converter import build_voc

import numpy as np


def calculate_nme(pd_land, gt_land, left_eye=0, right_eye=1):
    """
    实现计算人脸对齐的标准化平均误差(NME)。NME通常使用预测的关键点与真实关键点之间的欧氏距离
    :param pd_land: (n,2)
    :param gt_land: (n,2)
    :param left_eye: 左眼
    :param right_eye: 右眼
    :return:
    """
    # pd_land = np.array(pd_land, dtype=np.float32)
    # gt_land = np.array(gt_land, dtype=np.float32)
    # 计算标准化因子
    norm = np.linalg.norm(gt_land[left_eye] - gt_land[right_eye], axis=-1)
    # 计算每个关键点的欧氏距离误差
    l2 = np.linalg.norm(pd_land - gt_land, axis=-1)
    # 计算NME
    nme = np.mean(l2) / norm
    return nme


def calculate_nme_batch(pd_lands, gt_lands, left_eye=0, right_eye=1):
    """
    实现计算人脸对齐的标准化平均误差(NME)。NME通常使用预测的关键点与真实关键点之间的欧氏距离
    :param pd_lands: (b,n,2)
    :param gt_lands: (b,n,2)
    :param left_eye: 左眼
    :param right_eye:右眼
    :return:
    """
    # pd_lands = np.array(pd_lands, dtype=np.float32)
    # gt_lands = np.array(gt_lands, dtype=np.float32)
    # 计算标准化因子
    norm = np.linalg.norm(gt_lands[:, left_eye] - gt_lands[:, right_eye], axis=-1)
    # 计算每个关键点的欧氏距离误差
    l2 = np.linalg.norm(pd_lands - gt_lands, axis=-1)
    # 计算NME
    nme = np.mean(l2, axis=-1) / norm
    return nme


def str2number(x: str | int | float):
    """
    :param x:
    :return:
    """
    # 如果已经是数值类型，直接处理
    if isinstance(x, (int, float)):
        return int(x) if isinstance(x, int) or x.is_integer() else x
    # 处理字符串类型
    if isinstance(x, str):
        if not x: return x  # 处理空字符串
        x = x.strip()  # 去除首尾空格
        try:
            x = int(x)
        except Exception as e:
            try:
                x = float(x)
            except Exception as e:
                pass
    return x


if __name__ == '__main__':
    file = "/home/PKing/nasdata/Detector/Landmark/dataset/dataset-v2/dataset/test/list.txt"
    with time_utils.Performance("read_data1") as p:
        data1 = file_utils.read_data(file, split=" ")

    with time_utils.Performance("read_data2") as p:
        data2 = file_utils.read_data(file, split=" ")
    data = str2number(100)
    # print(type(data), data)
