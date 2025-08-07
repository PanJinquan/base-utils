# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2025-07-22 10:40:45
# @Brief  :
# --------------------------------------------------------
"""
import time

import cv2
import numpy as np
import re
from pybaseutils import text_utils, image_utils


def draw_rectangle(image, boxes, color=(255, 0, 0), thickness=10):
    """
    :param image: BGR image
    :param boxes: 矩形框[(xmin,ymin,xmax,ymax), ...]
    :param color: (b,g,r)或者(b,g,r,a) 其中a是透明度
    :return:
    """
    # 创建一个与图像大小相同的透明层
    bgim = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(bgim)
    # 在透明层上绘制半透明矩形
    for box in boxes:
        draw.rectangle(box, fill=color, outline=color, width=thickness)
    # 将透明层与原图像合并
    image = Image.alpha_composite(image.convert('RGBA'), bgim)
    image = image.convert('RGB')
    image = np.array(image)
    # image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


# 示例用法
if __name__ == "__main__":
    from PIL import Image, ImageDraw

    num = [929, 984, 933, 901, 1018, 902, 943, 1046, 1003, 837]
    print(len(num),sum(num))
