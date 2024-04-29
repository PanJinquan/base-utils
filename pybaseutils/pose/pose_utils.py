# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2023-09-05 10:23:50
    @Brief  :
"""
import numpy as np
import numbers
from pybaseutils import image_utils, geometry_tools, json_utils


def extend_line(p1, p2, scale=(1.0, 1.0)):
    p3 = (p2[0] + (p2[0] - p1[0]) * scale[0], p2[1] + (p2[1] - p1[1]) * scale[1])
    return p3


def extend_box(box, scale=[1.0, 1.0]):
    if len(box) == 0: return box
    box = image_utils.extend_xyxy(xyxy=[box], scale=scale, valid_range=[])[0]
    return box


def distance(p1, p2):
    d = geometry_tools.compute_distance(p1, p2)
    return d


def points2box(points):
    """
    将多边形轮廓转转为矩形框
    :param points: shape is (num_point,2)
    """
    if len(points) == 0: return []
    if not isinstance(points, np.ndarray): points = np.asarray(points)
    xmin = min(points[:, 0])
    ymin = min(points[:, 1])
    xmax = max(points[:, 0])
    ymax = max(points[:, 1])
    box = [xmin, ymin, xmax, ymax]
    return box


def create_box_from_point(c, r):
    """
    :param c: 中心点
    :param r: 半径(rw,rh)
    :return: (xmin,ymin,xmax,ymax)
    """
    if isinstance(r, numbers.Number): r = (r, r)
    return [c[0] - r[0], c[1] - r[1], c[0] + r[0], c[1] + r[1]]


def cal_iou(box1, box2):
    """
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
