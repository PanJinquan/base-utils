# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  : https://blog.csdn.net/qdPython/article/details/121381363
"""
import cv2
import random
import types
import numpy as np
from pybaseutils import image_utils, file_utils
import cv2


def resize_image_padding(image, size, color=(0, 0, 0), interpolation=cv2.INTER_LINEAR):
    """
    为保证图像无形变，需要进行等比例缩放和填充
    :param image:
    :param color: 短边进行填充的color value
    :return:
    """
    h, w = image.shape[:2]
    if w / h > size[0] / size[1] > 0:
        dsize = (size[0], None)
    else:
        dsize = (None, size[1])
    image = image_utils.resize_image(image, size=dsize, interpolation=interpolation)
    image = image_utils.center_crop_padding(image, crop_size=size, color=color)
    return image


if __name__ == '__main__':
    image_file = "/home/PKing/Downloads/test.jpg"
    src = cv2.imread(image_file)
    # src = cv2.resize(src, dsize=(200, 60))
    src = cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE)
    print(src, src.shape)
    image_utils.cv_show_image("src", src, delay=10)
    dsize = (500, 200)
    dst1 = image_utils.resize_image_padding(src, size=(dsize[0], dsize[1]))
    dst2 = image_utils.resize_image_padding(src, size=(dsize[1], dsize[0]))
    print("dst1", dst1.shape)
    print("dst2", dst2.shape)
    image_utils.cv_show_image("dst1", dst1, delay=10)
    image_utils.cv_show_image("dst2", dst2, delay=0)
