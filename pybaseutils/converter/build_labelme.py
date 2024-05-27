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
from tqdm import tqdm
from pybaseutils import image_utils, file_utils, json_utils


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


def del_labelme_imagedata(anno_dir):
    """
    删除labelme标注文件的imageData字段
    :param anno_dir:
    :return:
    """
    file_list = file_utils.get_files_lists(anno_dir, postfix=["*.json"])
    for anno_file in tqdm(file_list):
        data_info = json_utils.read_json_data(anno_file)
        data_info["imageData"] = None
        json_utils.write_json_path(anno_file, data_info)


def copy_labelme_files(image_dir, anno_dir, out_root):
    """
    复制labelme标注文件和图片文件
    :param image_dir:
    :param anno_dir:
    :param out_root:
    :return:
    """
    json_list = file_utils.get_files_list(anno_dir, postfix=["*.json"])
    out_images = file_utils.create_dir(out_root, "images")
    out_json = file_utils.create_dir(out_root, "json")
    for json_file in tqdm(json_list):
        json_data = json_utils.read_json_data(json_file)
        image_name = json_data['imagePath']
        shapes = json_data.get('shapes', [])
        image_file = os.path.join(image_dir, image_name)
        if len(shapes) > 0 and os.path.exists(image_file):
            file_utils.copy_file_to_dir(image_file, out_images)
            file_utils.copy_file_to_dir(json_file, out_json)
        else:
            print("bad json file:{}".format(json_file))
