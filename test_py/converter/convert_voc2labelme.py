# -*- coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-10-12 16:21:55
    @Brief  :
"""
import os
import numpy as np
import cv2
from pybaseutils.converter import build_voc, build_labelme, convert_voc2labelme

if __name__ == "__main__":
    """
    Bus       # 公共汽车
    Microbus  # 微型客车
    Minivan   # 小型货车
    SUV       # SUV车
    Sedan     # 轿车
    Truck     # 卡车
    """
    data_root = "/home/PKing/nasdata/tmp/tmp/pen/dataset-pen-finger-tip-005/train"
    out_root = data_root
    class_dict = {"car": "car",
                  "person": "pedestrian",
                  "bicycle": "bike",
                  "truck": "truck",
                  "motorcycle": "bike",
                  "bus": "bus"
                  }
    convert_voc2labelme.convert_voc2labelme(data_root=data_root,
                                            out_root=out_root,
                                            prefix="",
                                            class_name=None,
                                            class_dict={},
                                            shape_type="rotation",
                                            vis=False)
