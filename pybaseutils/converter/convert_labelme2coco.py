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

    def __init__(self, image_dir, anno_dir, filename="", init_id=None):
        """
        :param image_dir: 图片目录(*.json)
        :param anno_dir:  标注文件目录
        :param init_id: 初始的image_id,if None,will reset to current time
        """
        super(Labelme2COCO, self).__init__(init_id=init_id)
        print(anno_dir)
        print(image_dir)
        self.anno_dir = anno_dir
        self.image_dir = image_dir
        self.labelme = parser_labelme.LabelMeDataset(filename=filename, data_root=None, image_dir=image_dir,
                                                     anno_dir=anno_dir, class_name=None, use_rgb=False,
                                                     shuffle=False, check=False, )

    def build_keypoints_dataset(self, save_file, num_joints, class_name, kps_name=[], skeleton=[], only_kps=True):
        """
        构建COCO的关键点检测数据集
        :param save_file: string 输出COCO格式的文件
        :param num_joints: int 关键点个数
        :param class_name: list 目标关键点名称,仅支持单个类别,如['person']
        :param kps_name: 关键点的名称
        :param skeleton: 关键点连接点
        :return:
        """
        assert len(class_name) == 1  # 目前仅仅支持单个类别
        for index in tqdm(range(len(self.labelme.image_ids))):
            image_id = self.labelme.index2id(index)
            image_file, anno_file, image_id = self.labelme.get_image_anno_file(image_id)
            annotation, width, height = self.labelme.load_annotations(anno_file)
            if not annotation: continue
            objects = self.labelme.get_keypoint_object(annotation, width, height, class_name=class_name)
            if not objects: continue
            filename = os.path.basename(image_file)
            labels, boxes, contours, keypoints = [], [], [], []
            for group_id, object in objects.items():
                label = object["labels"]
                box = object["boxes"]
                segs = [object['segs']]
                kpts = json_utils.get_value(object, ["keypoints"], default={})
                if only_kps and not kpts:
                    continue
                kpts = self.get_keypoints_info(kpts, width, height, num_joints)
                labels.append(label)
                boxes.append(box)
                contours.append(segs)
                keypoints.append(kpts)
            info = {"boxes": boxes, "labels": labels, "contours": contours, "keypoints": keypoints}
            self.addObjects(filename, info, width, height, num_joints)
        # 设置关键点的名称和skeleton
        self.set_keypoints_category(kps_name=kps_name, skeleton=skeleton, cat_id=0)
        build_coco.COCOTools.check_coco(self.coco)
        self.save_coco(save_file)

    def build_instances_dataset(self, save_file, class_name=[]):
        """
        构建COCO的目标检测和实例分割数据集
        :param save_file: 输出COCO格式的文件
        :param class_name: 只选择的类别转换,默认全部
        :return:
        """
        for index in tqdm(range(len(self.labelme.image_ids))):
            image_id = self.labelme.index2id(index)
            # image_id = 'test3.png'
            image_file, anno_file, image_id = self.labelme.get_image_anno_file(image_id)
            annotation, width, height = self.labelme.load_annotations(anno_file)
            if not annotation: continue
            objects = self.labelme.get_instance_object(annotation, width, height, class_name=class_name)
            if not objects: continue
            filename = os.path.basename(image_file)
            labels, boxes, contours = [], [], []
            for group_id, object in objects.items():
                label = object["labels"]
                box = object["boxes"]
                segs = [object['segs']]
                labels.append(label)
                boxes.append(box)
                contours.append(segs)
            info = {"boxes": boxes, "labels": labels, "contours": contours}
            self.addObjects(filename, info, width, height, num_joints=0)
        build_coco.COCOTools.check_coco(self.coco)
        self.save_coco(save_file)

    def get_keypoints_info(self, keypoints: dict, width, height, num_joints):
        """
        keypoints=num_joints*3,x,y,visibility
        keypoints关节点的格式 : [x_1, y_1, v_1,...,x_k, y_k, v_k]
        其中x,y为Keypoint的坐标，v为可见标志
            v = 0 : 未标注点
            v = 1 : 标注了但是图像中不可见（例如遮挡）
            v = 2 : 标注了并图像可见
        实际预测时，不要求预测每个关节点的可见性
        :param keypoints:
        :param num_joints: 关键点个数
        :param width: 图像宽度
        :param height: 图像长度
        :return:
        """
        if len(keypoints) == 0: return []
        kps = np.zeros(shape=(num_joints, 3), dtype=np.int32)
        for i, v in keypoints.items():
            kps[i, :] = v + [2]  # (x,y,v=2)
        kps[:, 0] = np.clip(kps[:, 0], 0, width - 1)
        kps[:, 1] = np.clip(kps[:, 1], 0, height - 1)
        kps = kps.reshape(-1).tolist()
        return kps

    def save_coco(self, json_file):
        """保存COCO数据集"""
        super(Labelme2COCO, self).save_coco(json_file)


def demo():
    kps_name = ["p0", "p1", "p2", "p3", "p4"]
    skeleton = [[0, 2], [2, 1], [2, 3], [3, 4]]
    image_dir = "/media/PKing/新加卷1/SDK/base-utils/data/coco/JPEGImages"
    anno_dir = "/media/PKing/新加卷1/SDK/base-utils/data/coco/json"
    coco_ins_file = os.path.join(os.path.dirname(image_dir), "coco_ins.json")
    coco_kps_file = os.path.join(os.path.dirname(image_dir), "coco_kps.json")
    build = Labelme2COCO(image_dir=image_dir, anno_dir=anno_dir, init_id=None)
    # build.build_keypoints_dataset(coco_kps_file,
    #                               class_name=["person"],
    #                               kps_name=kps_name,
    #                               skeleton=skeleton,
    #                               num_joints=5)
    build.build_instances_dataset(coco_ins_file)


if __name__ == '__main__':
    demo()
