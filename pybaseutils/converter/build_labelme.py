# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-09-05 19:48:52
    @Brief  :
"""
import os
import numpy as np
import cv2
from pybaseutils import image_utils, file_utils, coords_utils


def maker_labelme(json_file, points, labels, image_name, image_size, image_bs64=None):
    """
    制作label数据格式
    :param json_file: 保存json文件路径
    :param points: (num_labels,num_points,2), points = image_utils.boxes2polygons(boxes)
    :param labels: (num_labels,)
    :param image_name: 图片名称，如果存在则进行拷贝到json_file同一级目录
    :param image_size: (W,H)
    :param image_bs64: 图片base64编码，可为None
    :return:
    """
    assert len(points) == len(labels)
    file_utils.create_file_path(json_file)
    shapes = []
    for point, label in zip(points, labels):
        # point = [[x1,y1],[x2,y2],...,[xn,yn]]
        if isinstance(point, np.ndarray): point = point.tolist()
        if not isinstance(point[0], list): point = [point]
        item = {"label": label, "line_color": None, "fill_color": None,
                "points": point, "shape_type": "polygon", "flags": {}}
        shapes.append(item)
    data = {
        "version": "3.16.7", "flags": {},
        "shapes": shapes,
        "lineColor": [0, 255, 0, 128],
        "fillColor": [255, 0, 0, 128],
        "imagePath": os.path.basename(image_name),
        "imageData": image_bs64,
        "imageHeight": image_size[1],
        "imageWidth": image_size[0]
    }
    if os.path.exists(image_name): file_utils.copy_file_to_dir(image_name, os.path.dirname(json_file))
    file_utils.write_json_path(json_file, data)
    return data
