# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project:
# @Author : panjq
# @Date   : 2020-02-12 18:28:16
# @url    :
# --------------------------------------------------------
"""

import argparse
import sys
import os
import numpy as np
import json
import cv2
import time
from tqdm import tqdm
from pybaseutils import file_utils, json_utils, image_utils
from pybaseutils.dataloader import parser_labelme
from pybaseutils.converter import build_coco


class Labelme2YOLO(object):
    """Convert Labelme to COCO dataset format"""

    def __init__(self, image_dir, anno_dir, init_id=None):
        """
        :param image_dir: 图片目录(*.json)
        :param anno_dir:  标注文件目录
        :param init_id: 初始的image_id,if None,will reset to current time
        """
        print(anno_dir)
        print(image_dir)
        self.anno_dir = anno_dir
        self.image_dir = image_dir
        self.labelme = parser_labelme.LabelMeDataset(filename=None, data_root=None, image_dir=image_dir,
                                                     anno_dir=anno_dir, class_name=None, use_rgb=False,
                                                     shuffle=False, check=False, )

    def build_instance_dataset(self, out_labels, class_name: list = []):
        """
        构建COCO的目标检测和实例分割数据集
        :param class_name: 只选择的类别转换,默认全部
        :return: 
        """
        for i in tqdm(range(len(self.labelme.image_ids))):
            image_id = self.labelme.index2id(i)
            image_file, anno_file, image_id = self.labelme.get_image_anno_file(image_id)
            annotation, width, height = self.labelme.load_annotations(anno_file)
            if not annotation: continue
            objects = self.labelme.get_instance_object(annotation, width, height, class_name=class_name)
            contents = []
            for group_id, object in objects.items():
                name = object["labels"]
                box = object['boxes'] / (width, height, width, height)
                seg = object['segs'] / (width, height)
                box = box.tolist()
                seg = seg.reshape(-1).tolist()
                if name in class_name:
                    label = class_name.index(name)
                    c = [label] + seg
                    contents.append(c)
            if len(contents) > 0:
                format = os.path.basename(image_file).split(".")[-1]
                image_id = os.path.basename(image_file)[:-len(format) - 1]
                text_path = file_utils.create_dir(out_labels, None, "{}.txt".format(image_id))
                file_utils.write_data(text_path, contents, split=" ")


def demo_for_yolo():
    class_name = ["other", "car", "person"]
    # VOC数据目录
    data_root = "/media/PKing/新加卷1/SDK/base-utils/data/coco"
    image_dir = "/media/PKing/新加卷1/SDK/base-utils/data/coco/JPEGImages"
    anno_dir = "/media/PKing/新加卷1/SDK/base-utils/data/coco/json"
    # data_root = "/path/to/VOC2007"
    # 保存输出yolo格式数据目录
    out_text_dir = os.path.join(data_root, "labels")
    build = Labelme2YOLO(image_dir=image_dir, anno_dir=anno_dir, init_id=None)
    build.build_instance_dataset(out_text_dir, class_name=class_name)


if __name__ == '__main__':
    demo_for_yolo()
