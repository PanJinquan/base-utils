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


@time_utils.performance("iou_calculation1")
def iou_calculation1(mask1, mask2):
    iou = image_utils.get_mask_iou(mask1, mask2)
    return iou


@time_utils.performance("iou_calculation2")
def iou_calculation2(mask1, mask2):
    """
    计算两个Mask的IOU
    :param mask1:
    :param mask2:
    :param binarize:
    :return:
    """
    iou = image_utils.get_mask_iou1(mask1, mask2)
    return iou


@time_utils.performance("iou_calculation3")
def iou_calculation3(mask1, mask2):
    """
    计算两个Mask的IOU
    :param mask1:
    :param mask2:
    :param binarize:
    :return:
    """
    inter = cv2.bitwise_and(mask1, mask2)  # 交集
    union = cv2.bitwise_or(mask1, mask2)  # 并集
    or_area = np.sum(np.float32(np.greater(inter, 0)))
    and_area = np.sum(np.float32(np.greater(union, 0)))
    or_area = np.sum(np.greater(inter, 0))
    and_area = np.sum(np.greater(union, 0))

    union_area = np.sum(union > 0)  # union>0
    inter_area = np.sum(inter > 0)  # inter>0
    iou = inter_area / max(union_area, 1e-8)
    return iou


def performance(mask1, mask2, max_iter=10):
    for i in range(max_iter):
        iou1 = iou_calculation1(mask1=mask1, mask2=mask2)
        iou2 = iou_calculation2(mask1=mask1, mask2=mask2)
        iou3 = iou_calculation3(mask1=mask1, mask2=mask2)
        print("=======" * 10)
    return iou1, iou2, iou3


if __name__ == "__main__":
    image_file = "test.png"
    image_file = "/home/dm/nasdata/release/handwriting/daip-calligraphy-hard/calligraphy-hard-evaluation/test/笔画缺失1/摘.png"
    image = image_utils.read_image(image_file, size=(2000, 2000))
    mask1 = image_utils.get_image_mask(image, inv=False)
    mask2 = image_utils.get_image_mask(image, inv=False)
    h, w = mask2.shape[:2]
    mask2[0:h // 2] = 0
    iou1, iou2, iou3 = performance(mask1, mask2)
    print("iou:{}".format([iou1, iou2, iou3]))
    image_utils.cv_show_image("mask1", mask1, delay=1)
    image_utils.cv_show_image("mask2", mask2, delay=0)
