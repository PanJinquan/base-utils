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


class Labelme2COCO(build_coco.COCOBuilder):
    """Convert Labelme to COCO dataset format"""

    def __init__(self, anno_dir, image_dir, class_name=[], init_id=None, use_kpt=False, kpt_names=[],
                 kpt_shape=[]):
        """
        :param anno_dir:
        :param image_dir:
        :param class_name:
        :param init_id:
        :param use_kpt:
        :param kpt_names: 关键点名称列表，
                          当kpt_names.ndim=1时，要求所有类别且所有目标的关键点名称必须一致
                          当kpt_names.ndim=2时，仅要求相同类别目标的关键点名称必须一致，且len(kpt_names)==len(self.class_name)
        :param kpt_shape: 关键点的维度，默认是(17, 3)，对于coco-person,有17个关键点，3表示(x,y,v)）
        """
        super(Labelme2COCO, self).__init__(init_id=init_id)
        print(anno_dir)
        print(image_dir)
        self.anno_dir = anno_dir
        self.image_dir = image_dir
        self.kpt_names = kpt_names
        self.kpt_shape = kpt_shape
        self.labelme = parser_labelme.LabelMeDataset(anno_dir=anno_dir,
                                                     image_dir=image_dir,
                                                     class_name=class_name,
                                                     check_kpt=True,
                                                     use_kpt=use_kpt,
                                                     kpt_names=kpt_names,
                                                     kpt_shape=kpt_shape,
                                                     use_rgb=False,
                                                     shuffle=False,
                                                     check=False
                                                     )

    def build_instances_dataset(self, save_file, ):
        """
        构建COCO的目标检测和实例分割数据集
        :param save_file: 输出COCO格式的文件
        :return:
        """
        for index in tqdm(range(len(self.labelme))):
            data_info = self.labelme.__getitem__(index)
            contours = [[c] for c in data_info["points"]]
            info = {"boxes": data_info["boxes"], "labels": data_info["names"],
                    "contours": contours}
            width, height = data_info["size"]
            filename = data_info["image_file"]
            self.addObjects(filename, info, width, height, num_joints=0)
        build_coco.COCOTools.check_coco(self.coco)
        self.save_coco(save_file)

    def build_keypoints_dataset(self, save_file, skeleton=[]):
        """
        构建COCO的目标检测和实例分割数据集
        :param save_file: 输出COCO格式的文件
        :return:
        """
        for index in tqdm(range(len(self.labelme))):
            data_info = self.labelme.__getitem__(index)
            contours = [[c] for c in data_info["points"]]
            keypoints = [c.reshape(-1) for c in data_info["keypoints"]]
            info = {"boxes": data_info["boxes"], "labels": data_info["names"],
                    "contours": contours, "keypoints": keypoints}
            width, height = data_info["size"]
            filename = data_info["image_file"]
            self.addObjects(filename, info, width, height)
        # 设置关键点的名称和skeleton
        self.set_keypoints_category(kps_name=self.kpt_names[0], skeleton=skeleton, cat_id=0)
        build_coco.COCOTools.check_coco(self.coco)
        self.save_coco(save_file)

    def save_coco(self, json_file):
        """保存COCO数据集"""
        super(Labelme2COCO, self).save_coco(json_file)


def demo():
    class_name = ["person"]
    kpt_names = [["p0", "p1", "p2", "p3", "p4"]]
    skeleton = [[0, 2], [2, 1], [2, 3], [3, 4]]
    kpt_shape = [5, 3]
    anno_dir = "/media/PKing/dev2/SDK/base-utils/data/labelme/images"
    coco_ins_file = os.path.join(os.path.dirname(anno_dir), "coco_ins.json")
    coco_kps_file = os.path.join(os.path.dirname(anno_dir), "coco_kps.json")
    build = Labelme2COCO(image_dir=anno_dir,
                         anno_dir=anno_dir,
                         class_name=class_name,
                         use_kpt=True,
                         kpt_names=kpt_names,
                         kpt_shape=kpt_shape
                         )
    # TODO 构建关键点检测数据集
    build.build_keypoints_dataset(coco_kps_file, skeleton=skeleton)
    # TODO 构建实例分割数据集
    # build.build_instances_dataset(coco_ins_file)


if __name__ == '__main__':
    demo()
