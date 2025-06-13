# -*- coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail :
    @Date   : 2024-05-23 11:24:37
    @Brief  : Series是一维数据结构，DataFrame二维表格结构，由多个Series组成（每列是一个Series）
"""
import os

import cv2
import numpy as np
from tqdm import tqdm
from pybaseutils import file_utils, image_utils, numpy_utils, pandas_utils, json_utils
from pybaseutils.cvutils import corner_utils
from pybaseutils.dataloader import parser_labelme
from pybaseutils.converter import build_labelme
from scipy.spatial.distance import cdist
import hashlib
import pandas as pd
import nltk
from rich import print_json


def cal_iou(box1, box2):
    """
    计算IOU=交集(A,B)/并集(A,B)
    计算IOM=交集(A,B)/最小集(A,B)
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return:
    """
    if len(box1) == 0: return 0
    if len(box2) == 0: return 0
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    # 计算每个矩形的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # C的面积
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # G的面积

    # 计算相交矩形
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    area = w * h  # C∩G的面积
    iou = area / (s1 + s2 - area)
    return iou


def cal_iom(box1, box2):
    """
    计算IOU=交集(A,B)/并集(A,B)
    计算IOM=交集(A,B)/最小集(A,B)
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return:
    """
    if len(box1) == 0: return 0
    if len(box2) == 0: return 0
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    # 计算每个矩形的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # C的面积
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # G的面积

    # 计算相交矩形
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    area = w * h  # C∩G的面积
    iou = area / min(s1, s2)
    return iou


def targets_overlap(obj_info1: dict, obj_info2: dict, score_th=0.1, iou_th=0.0):
    boxes1 = obj_info1['boxes']
    boxes2 = obj_info2['boxes']
    ious = image_utils.get_boxes_iom(boxes1, boxes2)
    output = []
    for i in range(len(boxes1)):
        obj1 = {k: v[i] if isinstance(v, list) else v for k, v in obj_info1.items()}
        obj2 = []
        for j in range(len(boxes2)):
            # iou = image_utils.get_box_iou(boxes1[i], boxes2[j])
            iou = image_utils.get_box_iom(boxes1[i], boxes2[j])
            item = {k: v[j] if isinstance(v, list) else v for k, v in obj_info2.items()}
            item.update(iou=iou)
            obj2.append(item)
        output.append({"objects1": obj1, "objects2": obj2})
    return output


if __name__ == '__main__':
    #
    obj_info1 = {'boxes': [[10, 10, 50, 50],
                           [10, 10, 50, 50]],
                 "labels": ["A1", "A1"]}
    obj_info2 = {'boxes': [[20, 20, 40, 60],
                           [20, 20, 45, 60],
                           [20, 20, 50, 60]],
                 "labels": ["B2", "B2", "B2"]}
    output = targets_overlap(obj_info1, obj_info2, score_th=0.1, iou_th=0.0)
    print(json_utils.formatting(output))
