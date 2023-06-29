# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  :
"""

import numpy as np
from pybaseutils import image_utils
import cv2


def get_smoking_roi(xyxy, scale=(), cut=0.3):
    """
    获得吸烟检测区
    :param xyxy: shape is (num-boxes,4),box is (xmin,ymin,xmax,ymax)
    :param scale: boxes缩放大小
    :param cut: 裁剪比例
    :return:
    """
    up_boxes = []
    for i in range(len(xyxy)):
        xmin, ymin, xmax, ymax = xyxy[i]
        w, h = (xmax - xmin), (ymax - ymin)
        ymax = max(ymin + h * cut, ymin + w)
        up_boxes.append([xmin, ymin, xmax, ymax])
    up_boxes = np.asarray(up_boxes)
    if scale: up_boxes = image_utils.extend_xyxy(up_boxes, scale=scale)
    return up_boxes


if __name__ == '__main__':
    image_file = "/home/PKing/nasdata/dataset/tmp/smoking/sample.png"
    boxes = [[47, 52, 255, 420]]  # 人体检测框
    boxes = np.asarray(boxes)
    image = cv2.imread(image_file)
    up = get_smoking_roi(xyxy=boxes, scale=(), cut=0.3)  # 获得吸烟检测区
    image = image_utils.draw_image_boxes(image, boxes=boxes, color=(255, 0, 0))
    image = image_utils.draw_image_boxes(image, boxes=up, color=(0, 255, 0))
    image_utils.cv_show_image("image", image)
