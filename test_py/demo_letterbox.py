# -*- coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  : https://blog.csdn.net/qdPython/article/details/121381363
"""
import os
import cv2
import random
import types
import torch
from typing import List, Tuple, Dict
import numpy as np
from typing import Callable
from PIL import Image
from pybaseutils import image_utils, file_utils, text_utils, pandas_utils, json_utils, base64_utils
import numpy as np
import cv2

# 使用示例和测试函数
def test_letterbox():
    """测试Letterbox函数"""
    # 创建测试图像
    image_file = "../data/test_image/grid2.png"
    src = image_utils.read_image(image_file)
    dsize = (320, 320)
    spts = [(50, 50),(120, 120)]
    text = [i for i in range(len(spts))]
    src = image_utils.draw_points_texts(src, spts, texts=text, color=(255, 0, 0), thickness=4, fontScale=0.5)
    dst, dpts = image_utils.letterbox(src, dsize=dsize, src_pts=spts)
    dst = image_utils.draw_points_texts(dst, dpts, texts=text, color=(0, 255, 0), thickness=1, fontScale=0.5)
    ipts = image_utils.letterbox_inverse(dpts, ssize=(src.shape[1], src.shape[0]), dsize=(dst.shape[1], dst.shape[0]))
    src = image_utils.draw_points_texts(src, ipts, texts=text, color=(0, 0, 255), thickness=1, fontScale=0.5)
    # 可视化结果
    image_utils.show_image("dst", dst, delay=5)
    image_utils.show_image("src", src)
    return dst, dpts


if __name__ == "__main__":
    # 运行测试
    test_letterbox()
