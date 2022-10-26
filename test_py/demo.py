# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-07-27 15:23:24
    @Brief  :
"""
import json
from pybaseutils import font_utils, file_utils, video_utils, numpy_utils

if __name__ == "__main__":
    import cv2
    import numpy as np

    t = [0.01, 0.02, 0.03]  # m
    t = np.asarray(t)*1000
    p = np.asarray(t) + 0.5
    # MSE（均方误差）、RMSE （均方根误差）、MAE （平均绝对误差）
    mse, rmse, mae = numpy_utils.get_error(t, p)
    print(mse, rmse, mae)
