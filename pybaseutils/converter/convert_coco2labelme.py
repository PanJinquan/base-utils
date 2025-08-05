# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2025-04-29 09:13:09
# @Brief  :
# --------------------------------------------------------
"""
import os
import numpy as np
import cv2
from tqdm import tqdm
from pybaseutils.dataloader import parser_coco_det, parser_coco_ins
from pybaseutils.converter import build_voc, build_labelme
from pybaseutils import file_utils, image_utils


def convert_coco2labelme(filename, out_root, class_name=None, class_dict={}, prefix="", vis=True, **kwargs):
    """
    将COCO格式转换为VOC格式
    :param filename:
    :param class_name: 需要选择的类别，None表示全部
    :param out_root: out_root
    :param class_dict: 类别映射
    :param prefix: 提供文件名前缀，则重新进行重新命令
    """
    dataset = parser_coco_ins.CocoInstance(anno_file=filename,
                                           data_root=None,
                                           anno_dir=None,
                                           image_dir=None,
                                           class_name=class_name,
                                           transform=None,
                                           use_rgb=False,
                                           decode=False,
                                           shuffle=False)
    print("have num:{}".format(len(dataset)))
    for i in tqdm(range(len(dataset))):
        data_info = dataset.__getitem__(i)
        image, labels, segs = data_info["image"], data_info["labels"], data_info["segs"]
        points = [s[0] for s in segs]
        image_file = data_info["image_file"]
        labels = np.asarray(labels, np.int32).reshape(-1).tolist()
        names = [dataset.class_name[i] for i in labels]
        if class_dict: names = [class_dict.get(n, n) for n in names]
        h, w = image.shape[:2]
        if len(names) == 0 or image is None:
            print("Error:{}".format(image_file))
            continue
        image_name = os.path.basename(image_file)
        if prefix:
            _, postfix = file_utils.split_postfix(image_name)
            image_name = f"{prefix}_{i:0=5d}.{postfix}"
        image_id = image_name.split(".")[0]
        json_file = file_utils.create_dir(out_root, "images", f"{image_id}.json")
        file_path = file_utils.create_dir(out_root, "images", f"{image_name}")
        build_labelme.maker_labelme(json_file, points, names, image_name, image_size=(w, h), image_bs64=None)
        file_utils.copy_file(image_file, file_path)


if __name__ == "__main__":
    anno_file = "/home/PKing/nasdata/tmp/tmp/car-det/dataset/src/val/instance_val.json"
    out_root = "/home/PKing/nasdata/tmp/tmp/car-det/dataset/src/val/labelme"
    class_dict = {"Pedestrian": "pedestrian", "Car": "car", "Truck": "truck", "Tram": "bus",
                  "Cyclist": "bike", "Tricycle": "bike"}
    convert_coco2labelme(anno_file, out_root, class_name=[], class_dict=class_dict, prefix="image_2021", vis=False)
