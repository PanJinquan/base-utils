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
from pybaseutils import image_utils, file_utils
import xmltodict


def maker_cvat(xml_file, points, labels, image_name, image_size):
    """
    制作CVAT数据格式
    :param xml_file: 保存xml文件路径
    :param points: (num_labels,num_points,2), points = image_utils.boxes2polygons(boxes)
    :param labels: (num_labels,)
    :param image_name: 图片名称，如果存在则进行拷贝到json_file同一级目录
    :param image_size: (W,H)
    :return:
    """
    assert len(points) == len(labels)
    file_utils.create_file_path(xml_file)
    objects = []
    for idx, (point, label) in enumerate(zip(points, labels)):
        if isinstance(point, np.ndarray): point = point.tolist()
        if not isinstance(point[0], list): point = [point]
        pt = [{'x': p[0], 'y': p[1]} for p in point]
        item = {'name': label,
                'deleted': '0',
                'verified': '0',
                'occluded': 'no',
                'date': None,
                'id': idx,
                'parts': {'hasparts': None, 'ispartof': None},
                'polygon': {'pt': pt, 'username': None},
                'attributes': None}
        objects.append(item)
    data_info = {
        "annotation": {
            'filename': os.path.basename(image_name),
            'folder': "",
            'source': {'sourceImage': None, 'sourceAnnotation': 'Datumaro'},
            'imagesize': {'nrows': image_size[1], 'ncols': image_size[0]},
            'object': objects
        }}
    xml_info = xmltodict.unparse(data_info)
    with open(xml_file, 'w') as xml_file:
        xml_file.write(xml_info)
    return data_info
