# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-16 15:39:50
    @Brief  :
"""

import numpy as np
from pybaseutils import plot_utils, image_utils, file_utils


def plot_sigmoid(x=[], step=0.1, inv=False):
    x = np.arange(start=x[0], stop=x[1], step=step)
    y = 1 / (1 + np.exp(-x))
    return x, y


if __name__ == '__main__':
    file = "/home/dm/nasdata/dataset/csdn/plate/CCPD-master/CRNN-Plate-Recognition/data/province_count.json"
    data: dict = file_utils.read_json_data(file)
    y = list(data.values())
    x = list(data.keys())
    # x = [str(i) for i in range(len(y))]
    # plot_utils.plot_bar_text(x, y)
    plot_utils.plot_bar(x, y,xlabel="好X", ylabel="Y",)
