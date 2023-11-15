# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-10-12 16:21:55
    @Brief  :
"""
import sys
import os

sys.path.insert(0, os.getcwd())
import cv2
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
from pybaseutils.converter import convert_labelme2coco


class Labelme2COCO(convert_labelme2coco.Labelme2COCO):
    """Convert Labelme to COCO dataset format"""

    def __init__(self, image_dir, anno_dir, filename="", init_id=None):
        """
        :param image_dir: 图片目录(*.json)
        :param anno_dir:  标注文件目录
        :param init_id: 初始的image_id,if None,will reset to current time
        """
        super(Labelme2COCO, self).__init__(image_dir, anno_dir, filename=filename, init_id=init_id)

    def get_keypoint_object(self, annotation: list, w, h, class_name=[], use_group=True):
        """
        获得labelme关键点检测数据
        :param annotation:
        :param w:
        :param h:
        :param class_name:
        :param use_group: True : 表示数据使用可能存在多个实例，并使用group_id标识了不同实例，且label使用了整数index表示
                          False: 表示数据只有一个实例，未使用group_id，且label={name}{index}表示，如pen0,pen1
        :return:
        """
        if use_group:
            return self.labelme.get_keypoint_object(annotation, w=w, h=h, class_name=class_name)
        else:
            # return self.get_keypoint_object_from_pen(annotation, w=w, h=h, class_name=class_name)
            return self.get_keypoint_object_from_pen_tip(annotation, w=w, h=h, class_name=class_name)

    def get_keypoint_object_from_pen(self, annotation: list, w, h, class_name=[]):
        """
        获得labelme关键点检测数据
        :param annotation:
        :param w:
        :param h:
        :param class_name:
        :param use_group: True : 表示数据使用可能存在多个实例，并使用group_id标识了不同实例，且label使用了整数index表示
                          False: 表示数据只有一个实例，未使用group_id，且label={name}{index}表示，如pen0,pen1
        :return:
        """
        kpts = {}
        labels = []
        contours = []  # 所以点的集合
        for i, anno in enumerate(annotation):
            points = np.asarray(anno["points"], dtype=np.int32)
            label, index = file_utils.split_letters_and_numbers(anno["label"].lower(), join=True)
            if len(points) != 1: continue
            contours.append(points)
            labels.append(label)
            if file_utils.is_int(index) and label in class_name:
                kpts.update({int(index): points.tolist()[0]})
        contours = np.asarray(contours).reshape(-1, 2)  # 使用所有点计算boxes
        # contours = np.asarray(list(kpts.values()))  # 仅使用kpts点计算boxes
        objects = {}
        if len(kpts) == 2 and "pen" in labels and "finger" in labels:
            # if len(kpts) > 0 and "pen" in labels and "finger" in labels:
            contours[:, 0] = np.clip(contours[:, 0], 0, w - 1)
            contours[:, 1] = np.clip(contours[:, 1], 0, h - 1)
            boxes = image_utils.polygons2boxes([contours])
            boxes = image_utils.extend_xyxy_similar_square(boxes, weight=0.6)
            boxes = image_utils.extend_xyxy(boxes, scale=[1.1, 1.1], valid_range=(0, 0, w, h), fixed=True, use_max=True)
            objects[0] = {"labels": class_name[0], "boxes": boxes[0], "segs": contours, "keypoints": kpts}
        return objects

    def get_keypoint_object_from_pen_tip(self, annotation: list, w, h, class_name=[]):
        """
        获得labelme关键点检测数据
        :param annotation:
        :param w:
        :param h:
        :param class_name:
        :param use_group: True : 表示数据使用可能存在多个实例，并使用group_id标识了不同实例，且label使用了整数index表示
                          False: 表示数据只有一个实例，未使用group_id，且label={name}{index}表示，如pen0,pen1
        :return:
        """
        kpts = {}
        contours = []  # 所以点的集合
        target = "pen0"
        for i, anno in enumerate(annotation):
            points = np.asarray(anno["points"], dtype=np.int32)
            name = anno["label"].lower()
            label, index = file_utils.split_letters_and_numbers(anno["label"].lower(), join=True)
            if len(points) != 1: continue
            contours.append(points)
            if name == target:
                kpts.update({int(index): points.tolist()[0]})
        objects = {}
        if 0 in kpts and len(contours) > 0:
            pen_tip = np.asarray(contours).reshape(-1, 2)  # 使用所有点计算boxes
            pen_tip = image_utils.polygons2boxes([pen_tip])[0]
            baseline = max(pen_tip[2] - pen_tip[0], pen_tip[3] - pen_tip[1])
            baseline = max(baseline * 0.3, 80)
            point = kpts[0]
            if abs(pen_tip[3] - pen_tip[1]) > 2 and abs(pen_tip[3] - pen_tip[1]) > 2:
                rates = (pen_tip[2] - pen_tip[0]) / (pen_tip[3] - pen_tip[1])  # w/h
                center = [[point[0], point[1], baseline * rates / 2, baseline / 2]]
            else:
                center = [[point[0], point[1], baseline / 2, baseline / 2]]
            boxes = image_utils.cxcywh2xyxy(center)
            boxes = image_utils.extend_xyxy_similar_square(boxes, weight=0.6, valid_range=(0, 0, w, h))
            objects[0] = {"labels": "pen_tip", "boxes": boxes[0], "segs": contours, "keypoints": kpts}
        return objects


def demo_for_person5():
    kps_name = ["p0", "p1", "p2", "p3", "p4"]
    skeleton = [[0, 2], [2, 1], [2, 3], [3, 4]]
    image_dir = "/media/PKing/新加卷1/SDK/base-utils/data/coco/JPEGImages"
    anno_dir = "/media/PKing/新加卷1/SDK/base-utils/data/coco/json"
    coco_ins_file = os.path.join(os.path.dirname(image_dir), "coco_ins.json")
    coco_kps_file = os.path.join(os.path.dirname(image_dir), "coco_kps.json")
    build = Labelme2COCO(image_dir=image_dir, anno_dir=anno_dir, init_id=None)
    build.build_keypoints_dataset(coco_kps_file,
                                  class_name=["person"],
                                  kps_name=kps_name,
                                  skeleton=skeleton,
                                  num_joints=5)
    # build.build_instances_dataset(coco_ins_file)


def demo_for_hand21():
    image_dir = "/media/PKing/新加卷1/SDK/base-utils/data/coco/JPEGImages"
    anno_dir = "/media/PKing/新加卷1/SDK/base-utils/data/coco/json"
    coco_ins_file = os.path.join(os.path.dirname(image_dir), "coco_ins.json")
    coco_kps_file = os.path.join(os.path.dirname(image_dir), "coco_kps.json")
    build = Labelme2COCO(image_dir=image_dir, anno_dir=anno_dir, init_id=None)
    # build.build_keypoints_dataset(coco_kps_file, class_name=["person"], num_joints=5)
    build.build_instances_dataset(coco_ins_file)


def demo_for_aije():
    """
    v2:{'绝缘鞋': 1, '身穿工作服': 2, '手': 3, '螺丝刀': 4, '验电笔': 5, '准备区域': 6, '钳形电流表': 7, '万用表': 8, '相序表': 9, '护目镜': 10, '绝缘手套': 11, '竹梯': 12, '表箱关': 13, '安全带': 14, '工作台': 15, '未穿工作服': 16, '其他鞋': 17, '相序表线头': 18, '万用表线头': 19, '电能表': 20, '电流互感器': 21, '接线盒': 22, '安全帽': 23}
    v4:{'电流互感器': 1, '表箱开': 2, '表箱关': 3, '竹梯': 4, '接线盒': 5, '电能表': 6, '安全帽': 7, '护目镜': 8, '绝缘手套': 9, '身穿工作服': 10, '工作台': 11, '准备区域': 12, '安全带': 13, '其他鞋': 14, '未穿工作服': 15, '万用表': 16, '螺丝刀': 17, '验电笔': 18, '绝缘鞋': 19, '万用表线头': 20, '相序表': 21, '相序表线头': 22, '钳形电流表': 23, '手': 24}
    :return:
    """
    filename = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-outdoor-det/dataset-v2/train.txt"
    image_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-outdoor-det/dataset-v2/JPEGImages"
    anno_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-outdoor-det/dataset-v2/json"
    coco_ins_file = filename.replace(".txt", "-coco.json")
    build = Labelme2COCO(image_dir=image_dir, anno_dir=anno_dir, filename=filename, init_id=None)
    build.build_instances_dataset(coco_ins_file)


def demo_for_pen():
    """
    v2:{'绝缘鞋': 1, '身穿工作服': 2, '手': 3, '螺丝刀': 4, '验电笔': 5, '准备区域': 6, '钳形电流表': 7, '万用表': 8, '相序表': 9, '护目镜': 10, '绝缘手套': 11, '竹梯': 12, '表箱关': 13, '安全带': 14, '工作台': 15, '未穿工作服': 16, '其他鞋': 17, '相序表线头': 18, '万用表线头': 19, '电能表': 20, '电流互感器': 21, '接线盒': 22, '安全帽': 23}
    v4:{'电流互感器': 1, '表箱开': 2, '表箱关': 3, '竹梯': 4, '接线盒': 5, '电能表': 6, '安全帽': 7, '护目镜': 8, '绝缘手套': 9, '身穿工作服': 10, '工作台': 11, '准备区域': 12, '安全带': 13, '其他鞋': 14, '未穿工作服': 15, '万用表': 16, '螺丝刀': 17, '验电笔': 18, '绝缘鞋': 19, '万用表线头': 20, '相序表': 21, '相序表线头': 22, '钳形电流表': 23, '手': 24}
    :return:
    """
    kps_name = ["0"]
    skeleton = [[0, 0]]
    # image_dir = "/home/PKing/nasdata/dataset/finger_keypoint/pen-v1-v9/images"
    # anno_dir = "/home/PKing/nasdata/dataset/finger_keypoint/pen-v1-v9/json"
    image_dir = "/home/PKing/nasdata/dataset/finger_keypoint/pen-test1-2/images"
    anno_dir = "/home/PKing/nasdata/dataset/finger_keypoint/pen-test1-2/json"
    class_name = ["pen"]
    coco_kps_file = os.path.join(os.path.dirname(image_dir), "coco_kps.json")
    build = Labelme2COCO(image_dir=image_dir, anno_dir=anno_dir, init_id=None)
    build.build_keypoints_dataset(coco_kps_file,
                                  class_name=class_name,
                                  kps_name=kps_name,
                                  skeleton=skeleton,
                                  num_joints=1,
                                  out_img=True,
                                  use_group=False,
                                  vis=True)


if __name__ == '__main__':
    # demo_for_person5()
    # demo_for_aije()
    demo_for_pen()
