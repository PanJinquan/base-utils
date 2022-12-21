# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-16 15:39:50
    @Brief  :
"""

import numpy as np
from pybaseutils import plot_utils,image_utils


def plot_sigmoid(x=[], step=0.1, inv=False):
    x = np.arange(start=x[0], stop=x[1], step=step)
    y = 1 / (1 + np.exp(-x))
    return x, y




if __name__ == "__main__":
    x = [-10, 10]
    x1, y1 = plot_sigmoid(x=x, step=0.1, inv=False)
    plot_utils.plot_multi_line(x_data_list=[x1], y_data_list=[y1])
