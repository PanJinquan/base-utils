# -*- coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail :
    @Date   : 2024-02-05 18:19:18
    @Brief  :
"""
import cv2
import numpy as np
from pybaseutils import file_utils, image_utils, plot_utils, numpy_utils


def gaussian_impulse(x, c=0, sigma=0.1):
    """
    高斯脉冲
    :param x: 输入数据x
    :param c: 脉冲中心点
    :param sigma:标准差，越小脉冲越尖锐
    :return: 生成高斯脉冲
    """
    y = np.exp(-((x - c) ** 2) / (2 * sigma ** 2))
    return y


def get_stage_times(cutline, stage_name, duration, offset=5):
    """
    获得环节的起止时间
    :param cutline:
    :param stage_name:
    :param duration:
    :param offset:
    :return:
    """
    assert len(stage_name) - len(cutline) == 1
    stage_time = {}
    cut_time = [0] + cutline + [duration]
    for i, T in enumerate(stage_name):
        start = max(0, cut_time[i] - offset)
        end = min(duration, cut_time[i + 1] + offset)
        stage_time[T] = dict(start=start, end=end)
    return stage_time, cut_time


if __name__ == '__main__':
    x1 = np.asarray([255, 245, 0, 1, 2], dtype=np.uint8)
    x2 = x1 - 10
    print(x1.dtype, x1)
    print(x2.dtype, x2)
