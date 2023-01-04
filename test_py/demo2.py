# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  :
"""
import cv2
import numpy as np
from pybaseutils import image_utils, file_utils, json_utils, time_utils


@time_utils.performance("get_mask_boundrect_low")
def get_mask_boundrect_low(mask):
    box = image_utils.get_mask_boundrect_low(mask, binarize=False)
    return box


@time_utils.performance("get_mask_boundrect")
def get_mask_boundrect(mask):
    box = image_utils.get_mask_boundrect(mask, binarize=False)
    return box


@time_utils.performance("get_mask_boundrect_cv")
def get_mask_boundrect_cv(mask):
    box = image_utils.get_mask_boundrect_cv(mask, binarize=False)
    return box


def performance(mask, func=None, max_iter=10):
    for i in range(max_iter):
        box = func(mask.copy())
    return box


if __name__ == "__main__":
    image_file = "test.png"
    image_file = "/home/dm/nasdata/release/handwriting/daip-calligraphy-hard/calligraphy-hard-evaluation/test/笔画缺失1/摘.png"
    image = image_utils.read_image(image_file)
    mask = image_utils.get_image_mask(image, inv=False)
    # mask = np.zeros(shape=(500, 500), dtype=np.uint8)
    box1 = performance(mask, func=get_mask_boundrect_cv)
    print("=======" * 10)
    box2 = performance(mask, func=get_mask_boundrect)
    print("=======" * 10)
    box3 = performance(mask, func=get_mask_boundrect_low)
    print("=======" * 10)

    print("box1:{}".format(box1))
    print("box2:{}".format(box2))
    print("box3:{}".format(box3))
    if len(box1) > 0: image = image_utils.draw_image_boxes(image, boxes_list=[box1], thickness=2)
    image_utils.cv_show_image("mask", mask, delay=1)
    image_utils.cv_show_image("image", image)
