# -*- coding: utf-8 -*-
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


def save_labelme(out_root, image_file, points, names, class_dict={}, group=None, image=None, prefix="",
                 index=0, vis=False, delay=0):
    """
    :param out_root:   输出根目录
    :param image_file: 图片路径
    :param points: 目标轮廓
    :param names:  目标名称
    :param class_dict: 需要映射的类别
    :param task:   任务类型:det,obb,seg,pose:
    :param image:  图像
    :param prefix: 前缀，如果提供，则重新命名
    :param index:  提供前缀需要重新名称
    :return:
    """
    if class_dict: names = [class_dict.get(n, n) for n in names]
    if image is None: image = cv2.imread(image_file)
    h, w = image.shape[:2]
    if image is None:
        print("Error: image is None,image_file={},names={}".format(image_file, names))
        return
    if vis:
        image = image_utils.draw_image_contours(image, points, texts=names, alpha=0.3)
        image = image_utils.show_image("image", image, delay=delay)
    image_name = os.path.basename(image_file)
    image_id, postfix = file_utils.split_postfix(image_name)
    if prefix:
        flag_ = file_utils.get_time(format="p") if index < 0 else f"{index:0=5d}"
        image_name = f"{prefix}_{flag_}.jpg"
    image_id, postfix = file_utils.split_postfix(image_name)
    json_file = file_utils.create_dir(out_root, "images", f"{image_id}.json")
    file_path = file_utils.create_dir(out_root, "images", f"{image_name}")
    maker_labelme(json_file, points, names, image_name, group=group, image_size=(w, h), image_bs64=None)
    if isinstance(image, np.ndarray):
        cv2.imwrite(file_path, image)
    else:
        file_utils.copy_file(image_file, file_path)
    if len(points) == 0:
        print("points is empty,file={}".format(file_path))


def maker_labelme(json_file, points, labels, image_name, image_size, group=None, image_bs64=None, keypoints=[]):
    """
    制作label数据格式
    :param json_file: 保存json文件路径
    :param points: (num_labels,num_points,2), points = image_utils.boxes2polygons(boxes)
    :param labels: (num_labels,)
    :param image_name: 图片名称，如果存在则进行拷贝到json_file同一级目录
    :param image_size: (W,H)
    :param image_bs64: 图片base64编码，可为None
    :param keypoints: [(N,3),(N,3),...],(x,y,conf),其3维度是置信度
    :return:
    """
    assert len(points) == len(labels)
    file_utils.create_file_path(json_file)
    shapes = []
    if isinstance(keypoints, np.ndarray): keypoints = keypoints.tolist()
    if group is None: group = [None] * len(points)
    for i in range(len(points)):
        # point = [[x1,y1],[x2,y2],...,[xn,yn]]
        point, label = points[i], labels[i]
        if isinstance(point, np.ndarray): point = point.tolist()
        if not isinstance(point[0], list): point = [point]
        kpts = keypoints[i] if keypoints else []
        item = {"label": label, "score": None, "keypoints": kpts, "line_color": None, "fill_color": None,
                "group_id": group[i], "points": point, "shape_type": "polygon", "flags": {}, "description": ""}
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
    file_utils.save_json(json_file, data)
    return data


def del_labelme_imagedata(anno_dir):
    """
    删除labelme标注文件的imageData字段
    :param anno_dir:
    :return:
    """
    file_list = file_utils.get_files_lists(anno_dir, postfix=["*.json"])
    for anno_file in tqdm(file_list):
        data_info = json_utils.load_json(anno_file)
        data_info["imageData"] = None
        json_utils.save_json(anno_file, data_info)


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
        json_data = json_utils.load_json(json_file)
        image_name = json_data['imagePath']
        shapes = json_data.get('shapes', [])
        image_file = os.path.join(image_dir, image_name)
        if len(shapes) > 0 and os.path.exists(image_file):
            file_utils.copy_file_to_dir(image_file, out_images)
            file_utils.copy_file_to_dir(json_file, out_json)
        else:
            print("bad json file:{}".format(json_file))
