# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  :
"""
import os
import cv2
import time
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
from pybaseutils import file_utils, image_utils, base64_utils, time_utils


@time_utils.performance("serialization")
def image_serialization(image):
    image = base64_utils.serialization(image)
    return image


@time_utils.performance("deserialization")
def image_deserialization(image):
    image = base64_utils.deserialization(image)
    return image


@time_utils.performance("resize")
def image_resize(image, dsize=(640, 640)):
    # image = cv2.resize(image, dsize=dsize)
    image = np.asarray(image, dtype=np.float32)
    image /= 255.0  # 0 - 255 to 0.0 - 1.0
    image = image_utils.image_boxes_resize_padding(image, dsize, color=(114, 114, 114))
    return image


def task_for_performance(image, max_iter=50):
    for i in range(max_iter):
        image = image_serialization(image.copy())
        image = image_deserialization(image)
        output = image_resize(image)
        print("input size:{},output size:{}".format(image.shape, output.shape))
        print("-----" * 10)


if __name__ == "__main__":
    image_file = "/home/dm/nasdata/release/handwriting/daip-calligraphy-hard/calligraphy-hard-detector/test/performance/test1.jpg"
    image = cv2.imread(image_file)
    dsize = (2000, 2000)
    r = 1.0
    dsize = (int(dsize[0] * r), int(dsize[1] * r))
    image = cv2.resize(image, dsize=tuple(dsize))
    # task_for_performance(image)
