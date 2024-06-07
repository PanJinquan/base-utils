# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail :
    @Date   : 2024-02-05 18:19:18
    @Brief  :
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
from pybaseutils import image_utils, file_utils, json_utils, base64_utils, time_utils
from pybaseutils.transforms import transform_utils
from pybaseutils.converter import build_voc, build_labelme
from pybaseutils.dataloader import parser_labelme


def read_image(image_file: str, use_rgb=True):
    """
    :param image_file:
    :param use_rgb:
    :return:
    """
    try:
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = image[:, :, 0:3]
        if use_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise Exception("empty image:{}".format(image_file))
    return image


@time_utils.performance()
def test_time1(image_file):
    image = cv2.imread(image_file, cv2.IMREAD_REDUCED_COLOR_2)
    # image = cv2.imread(image_file)
    print("image1 size:{}".format(image.shape))
    image_utils.cv_show_image("image", image)
    return image


@time_utils.performance()
def test_time2(image_file):
    image = cv2.imread(image_file)
    print("image2 size:{}".format(image.shape))
    image_utils.cv_show_image("image", image)
    return image


if __name__ == '__main__':
    import re

    text = "好，A,，？.,B1.2"
    repl = " "
    res = re.sub('[^\w\s]', repl, text)
    print(res)
    res = re.sub('[,，。/？；;:：#￥！!]', repl, text)
    text = text.strip()
    print(res)
