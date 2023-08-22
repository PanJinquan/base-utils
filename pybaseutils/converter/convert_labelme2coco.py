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

    def __init__(self, image_dir, anno_dir, init_id=None):
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
        self.labelme = parser_labelme.LabelMeDataset(filename=None, data_root=None, image_dir=image_dir,
                                                     anno_dir=anno_dir, class_name=None, use_rgb=False,
                                                     shuffle=False, check=False, )

    def build_keypoint_dataset(self, class_name=["person"]):
        """
        构建COCO的关键点检测数据集
        :param class_name: 目标关键点名称
        :return:
        """
        assert len(class_name) == 1  # 目前仅仅支持单个类别
        for index in tqdm(range(len(self.labelme.image_id))):
            image_id = self.labelme.index2id(index)
            image_file, anno_file, image_id = self.labelme.get_image_anno_file(image_id)
            annotation, width, height = self.labelme.load_annotations(anno_file)
            if not annotation: continue
            filename = os.path.basename(image_file)
            if filename in self.category_set:
                raise Exception('file_name duplicated')
            if filename not in self.image_set:
                image_size = [height, width]
                current_image_id = self.addImgItem(filename, image_size=image_size)
            else:
                raise Exception('duplicated image_dict: {}'.format(filename))
            objects = self.labelme.get_keypoint_object(annotation, width, height, class_name=class_name)
            for group_id, object in objects.items():
                name = object["labels"]
                box = object['boxes']
                seg = object['segs']
                if name not in self.category_set:
                    current_category_id = self.addCatItem(name)
                else:
                    current_category_id = self.category_set[name]
                if isinstance(box, np.ndarray): box = box.tolist()
                xmin, ymin, xmax, ymax = box
                rect = [xmin, ymin, xmax - xmin, ymax - ymin]
                # get segmentation info
                seg, area = self.get_segment_info([seg])
                keypoints = json_utils.get_value(object, ["keypoints"], default={})
                keypoints = self.get_keypoints_info(keypoints, width, height)
                self.addAnnoItem(current_image_id, current_category_id, rect, seg, area, keypoints=keypoints)
        build_coco.COCOTools.check_coco(self.coco)

    def build_instance_dataset(self, class_name=[]):
        """
        构建COCO的目标检测和实例分割数据集
        :param class_name: 只选择的类别转换,默认全部
        :return: 
        """
        for index in tqdm(range(len(self.labelme.image_id))):
            image_id = self.labelme.index2id(index)
            # image_id = 'test3.png'
            image_file, anno_file, image_id = self.labelme.get_image_anno_file(image_id)
            annotation, width, height = self.labelme.load_annotations(anno_file)
            if not annotation: continue
            filename = os.path.basename(image_file)
            if filename in self.category_set:
                raise Exception('file_name duplicated')
            if filename not in self.image_set:
                image_size = [height, width]
                current_image_id = self.addImgItem(filename, image_size=image_size)
            else:
                raise Exception('duplicated image_dict: {}'.format(filename))
            objects = self.labelme.get_instance_object(annotation, width, height, class_name=class_name)
            for group_id, object in objects.items():
                name = object["labels"]
                box = object['boxes']
                seg = object['segs']
                if name not in self.category_set:
                    current_category_id = self.addCatItem(name)
                else:
                    current_category_id = self.category_set[name]
                if isinstance(box, np.ndarray): box = box.tolist()
                xmin, ymin, xmax, ymax = box
                rect = [xmin, ymin, xmax - xmin, ymax - ymin]
                # get segmentation info
                seg, area = self.get_segment_info([seg])
                self.addAnnoItem(current_image_id, current_category_id, rect, seg, area, keypoints=[])
        build_coco.COCOTools.check_coco(self.coco)

    def get_keypoints_info(self, keypoints: dict, width, height):
        """
        keypoints=17*3,x,y,visibility
        keypoints关节点的格式 : [x_1, y_1, v_1,...,x_k, y_k, v_k]
        其中x,y为Keypoint的坐标，v为可见标志
            v = 0 : 未标注点
            v = 1 : 标注了但是图像中不可见（例如遮挡）
            v = 2 : 标注了并图像可见
        实际预测时，不要求预测每个关节点的可见性
        """
        kpts = np.zeros(shape=(17, 3), dtype=np.int32)
        for i, v in keypoints.items():
            kpts[i, :] = v + [2]  # (x,y,v=2)
        kpts[:, 0] = np.clip(kpts[:, 0], 0, width - 1)
        kpts[:, 1] = np.clip(kpts[:, 1], 0, height - 1)
        kpts = kpts.reshape(-1).tolist()
        return kpts

    def save_coco(self, json_file):
        """保存COCO数据集"""
        super(Labelme2COCO, self).save_coco(json_file)


def demo_for_voc():
    image_dir = "/media/PKing/新加卷1/SDK/base-utils/data/coco/JPEGImages"
    anno_dir = "/media/PKing/新加卷1/SDK/base-utils/data/coco/json"
    save_coco_file = os.path.join(os.path.dirname(image_dir), "person.json")
    build = Labelme2COCO(image_dir=image_dir, anno_dir=anno_dir, init_id=None)
    # build.build_keypoint_dataset()
    build.build_instance_dataset()
    build.save_coco(save_coco_file)


if __name__ == '__main__':
    demo_for_voc()
