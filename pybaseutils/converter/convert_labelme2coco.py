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
    """Convert Pascal VOC Dataset to COCO dataset format"""

    def __init__(self, image_dir, anno_dir, init_id=None):
        """
        :param anno_dir:  for voc `Annotations`
        :param image_dir: for voc `JPEGImages`,if image_dir=None ,will ignore checking image shape
        :param image_id: 初始的image_id,if None,will reset to currrent time
        """
        super(Labelme2COCO, self).__init__(init_id=init_id)
        print(anno_dir)
        print(image_dir)
        self.anno_dir = anno_dir
        self.image_dir = image_dir
        self.labelme = parser_labelme.LabelMeDataset(filename=None, data_root=None, image_dir=image_dir,
                                                     anno_dir=anno_dir, class_name=None, use_rgb=False,
                                                     shuffle=False, check=False, )

    def get_group_object(self, annotation: list, w, h, target="person"):
        objects = {}
        for anno in annotation:
            label = anno["label"].lower()
            points = np.asarray(anno["points"], dtype=np.int32)
            group_id = anno["group_id"] if anno["group_id"] else 0
            if target.lower() == label:
                segs = points
                segs[:, 0] = np.clip(segs[:, 0], 0, w - 1)
                segs[:, 1] = np.clip(segs[:, 1], 0, h - 1)
                box = image_utils.polygons2boxes([segs])[0]
                objects = json_utils.set_value(objects, key=[group_id],
                                               value={"labels": label, "boxes": box, "segs": segs})
            elif file_utils.is_int(label):
                keypoints: dict = json_utils.get_value(objects, [group_id, "keypoints"], default={})
                keypoints.update({int(label): points.tolist()[0]})
                objects = json_utils.set_value(objects, key=[group_id, "keypoints"], value=keypoints)
        return objects

    def build_dataset(self, class_dict={}):
        """构建从VOC到COCO的数据集"""
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
            objects = self.get_group_object(annotation, width, height)
            for group_id, object in objects.items():
                name = object["labels"]
                box = object['boxes']
                seg = object['segs']
                keypoints = json_utils.get_value(object, ["keypoints"], default=[])
                if class_dict and name in class_dict: name = class_dict[name]
                if name not in self.category_set:
                    current_category_id = self.addCatItem(name)
                else:
                    current_category_id = self.category_set[name]
                if isinstance(box, np.ndarray): box = box.tolist()
                xmin, ymin, xmax, ymax = box
                rect = [xmin, ymin, xmax - xmin, ymax - ymin]
                # get segmentation info
                seg, area = self.get_segment_info([seg])
                keypoints = self.get_keypoints_info(keypoints, width, height)
                self.addAnnoItem(current_image_id, current_category_id, rect, seg, area, keypoints=keypoints)
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
        kpts[:, 0] = np.clip(kpts[:, 0], 1, width - 1)
        kpts[:, 1] = np.clip(kpts[:, 1], 1, height - 1)
        kpts = kpts.reshape(-1).tolist()
        return kpts

    def save_coco(self, json_file):
        """保存COCO数据集"""
        super(Labelme2COCO, self).save_coco(json_file)


def demo_for_voc():
    image_dir = "/media/PKing/新加卷1/SDK/base-utils/data/person"
    anno_dir = "/media/PKing/新加卷1/SDK/base-utils/data/person"
    save_coco_file = os.path.join(os.path.dirname(image_dir), "person.json")
    build = Labelme2COCO(image_dir=image_dir, anno_dir=anno_dir, init_id=None)
    build.build_dataset()
    build.save_coco(save_coco_file)


if __name__ == '__main__':
    demo_for_voc()
