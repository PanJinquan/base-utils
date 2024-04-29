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
from pybaseutils import file_utils
from pybaseutils.dataloader import parser_voc
from pybaseutils.converter import build_coco


class VOC2COCO(build_coco.COCOBuilder):
    """Convert Pascal VOC Dataset to COCO dataset format"""

    def __init__(self, image_dir, anno_dir, seg_dir=None, init_id=None):
        """
        :param anno_dir:  for voc `Annotations`
        :param image_dir: for voc `JPEGImages`,if image_dir=None ,will ignore checking image shape
        :param seg_dir:   for voc `SegmentationObject`,if seg_dir=None,will ignore Segmentation Object
        :param image_id: 初始的image_id,if None,will reset to currrent time
        """
        super(VOC2COCO, self).__init__(init_id=init_id)
        print(anno_dir)
        print(image_dir)
        self.anno_dir = anno_dir
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.voc = parser_voc.VOCDataset(filename=None, data_root=None, image_dir=image_dir,
                                         anno_dir=anno_dir, seg_dir=seg_dir, class_name=None)

    def build_dataset(self, class_dict={}):
        """构建从VOC到COCO的数据集"""
        for index in tqdm(range(len(self.voc.image_ids))):
            image_id = self.voc.index2id(index)
            image_file, annotation_file = self.voc.get_image_anno_file(image_id)
            objects = self.voc.get_annotation(annotation_file)
            height, width = objects['height'], objects['width']
            boxes, labels = objects['boxes'], objects['labels']
            filename = os.path.basename(image_file)
            if len(boxes)==0: continue
            if filename in self.category_set:
                raise Exception('file_name duplicated')
            if filename not in self.image_set:
                image_size = [height, width]
                current_image_id = self.addImgItem(filename, image_size=image_size)
            else:
                raise Exception('duplicated image_dict: {}'.format(filename))
            for box, label in zip(boxes, labels):
                name = self.voc.class_name[label]
                if class_dict and name in class_dict: name = class_dict[name]
                if name not in self.category_set:
                    current_category_id = self.addCatItem(name)
                else:
                    current_category_id = self.category_set[name]
                if isinstance(box, np.ndarray): box = box.tolist()
                xmin, ymin, xmax, ymax = box
                rect = [xmin, ymin, xmax - xmin, ymax - ymin]
                # get segmentation info
                seg, area = self.voc.get_segment_info(filename, bbox=box)
                self.addAnnoItem(current_image_id, current_category_id, rect, seg, area, keypoints=[])
        build_coco.COCOTools.check_coco(self.coco)

    def save_coco(self, json_file):
        """保存COCO数据集"""
        super(VOC2COCO, self).save_coco(json_file)


def demo_for_voc():
    data_root = "/media/PKing/新加卷1/SDK/base-utils/data/VOC2007"
    image_dir = os.path.join(data_root, "JPEGImages")
    anno_dir = os.path.join(data_root, 'Annotations')  # 这是xml文所在的地址
    seg_dir = os.path.join(data_root, 'SegmentationObject')
    save_coco_file = os.path.join(data_root, "voc_coco_demo.json")
    build = VOC2COCO(image_dir=image_dir, anno_dir=anno_dir, seg_dir=seg_dir, init_id=None)
    build.build_dataset()
    build.save_coco(save_coco_file)


if __name__ == '__main__':
    demo_for_voc()
