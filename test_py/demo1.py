# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  :
"""
import asyncio
import time
import os
import cv2
import time
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
from pybaseutils import file_utils, image_utils, base64_utils, time_utils, json_utils


def image_rotation(image, angle, center=None, scale=1.0, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)):
    """
    图像旋转
    :param image:
    :param angle:
    :param center:
    :param scale:
    :param borderMode : 旋转边界填充方法
    :param borderValue: 旋转边界填充像素值
    :return:
    """
    h, w = image.shape[:2]
    if not center:
        # center = (w // 2, h // 2)
        center = (w / 2., h / 2.)
    mat = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, mat, dsize=(w, h), borderMode=borderMode, borderValue=borderValue)
    return rotated


if __name__ == "__main__":
    image_file = "/home/dm/nasdata/dataset-dmai/handwriting/grid-det/test.jpg"
    image = image_utils.read_image(image_file)
    image_utils.cv_show_image("image", image, delay=1)
    angle = -5
    image0 = image_rotation(image, angle=angle, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    image128 = image_rotation(image, angle=angle, borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))
    image3 = image_rotation(image, angle=angle, borderMode=cv2.BORDER_REPLICATE, borderValue=(128, 128, 128))
    image_utils.cv_show_image("BORDER_CONSTANT0", image0, delay=1)
    image_utils.cv_show_image("BORDER_CONSTANT128", image128, delay=1)
    image_utils.cv_show_image("BORDER_REPLICATE", image3, delay=0)
