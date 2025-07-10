# -*- coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-10-12 16:21:55
    @Brief  :
"""
import os
import numpy as np
import cv2
from tqdm import tqdm
from pybaseutils.converter import build_labelme
from pybaseutils.dataloader import parser_labelme
from pybaseutils import file_utils, image_utils


def convert_coco2labelme(anno_dir, out_root, class_name=None, class_dict={}, prefix="", vis=True, **kwargs):
    """
    将COCO格式转换为VOC格式
    :param filename:
    :param class_name: 需要选择的类别，None表示全部
    :param out_root: out_root
    :param class_dict: 类别映射
    :param prefix: 提供文件名前缀，则重新进行重新命令
    """
    dataset = parser_labelme.LabelMeDatasets(filename=None,
                                             data_root=None,
                                             anno_dir=anno_dir,
                                             image_dir=None,
                                             class_name=class_name,
                                             check=False,
                                             phase="val",
                                             shuffle=True)
    print("have num:{}".format(len(dataset)))
    for i in tqdm(range(len(dataset))):
        data_info = dataset.__getitem__(i)
        image, names, points = data_info["image"], data_info["names"], data_info["points"]
        image_file = data_info["image_file"]
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
    anno_file = "/home/PKing/nasdata/tmp/tmp/cat-det/dataset/video/car-dataset"
    out_root = "/home/PKing/nasdata/tmp/tmp/cat-det/dataset/video/labelme"
    convert_coco2labelme(anno_file, out_root, class_name=[], vis=False)
