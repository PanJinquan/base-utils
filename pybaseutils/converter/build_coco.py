# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project:
# @Author : panjq
# @Date   : 2020-02-12 18:28:16
# @url    :
# --------------------------------------------------------
"""

import sys
import os
import cv2
import time
import numpy as np
from tqdm import tqdm
from pybaseutils import file_utils


class COCOTools(object):
    """COCO Tools"""

    @staticmethod
    def get_categories_id(categories):
        """
        get categories id dict
        :param categories:
        :return: dict:{name:id}
        """
        assert categories, Exception("categories is empty")
        supercategorys = []
        categories_id = {}
        for item in categories:
            supercategory = item["supercategory"]
            name = item["name"]
            id = item["id"]
            categories_id[name] = id
        return categories_id

    @staticmethod
    def get_annotations_id(annotations):
        """
        get annotations id list
        :param annotations:
        :return: annotations id list
        """
        assert annotations, Exception("annotations is empty")
        annotations_id = []
        for item in annotations:
            id = item["id"]
            annotations_id.append(id)
        return annotations_id

    @staticmethod
    def get_images_id(images):
        """
        get image id list
        :param images:
        :return: images id list
        """
        assert images, Exception("images is empty")
        images_id = []
        for item in images:
            id = item["id"]
            images_id.append(id)
        return images_id

    @staticmethod
    def check_uniqueness(id_list: list, title="id"):
        """
        检测唯一性
        :return:
        """
        print("check {}".format(title))
        for i in id_list:
            n = id_list.count(i)
            assert n == 1, Exception("have same {}:{}".format(title, i))

    @staticmethod
    def check_coco(coco):
        """
        检测COCO合并后数据集的合法性
            检测1: 检测categories id唯一性
            检测2: 检测image id唯一性
            检测3: 检测annotations id唯一性
        :return:
        """
        categories_id = COCOTools.get_categories_id(coco["categories"])
        print("categories_id:{}".format(categories_id))
        categories_id = list(categories_id.values())
        COCOTools.check_uniqueness(categories_id, title="categories_id")

        image_id = COCOTools.get_images_id(coco["images"])
        COCOTools.check_uniqueness(image_id, title="image_id")

        annotations_id = COCOTools.get_annotations_id(coco["annotations"])
        COCOTools.check_uniqueness(annotations_id, title="annotations_id")
        print("have image:{}".format(len(image_id)))


class COCOBuilder():
    """Convert Pascal VOC Dataset to COCO dataset format"""

    def __init__(self, init_id=None):
        """
        :param anno_dir:  for voc `Annotations`
        :param image_dir: for voc `JPEGImages`,if image_dir=None ,will ignore checking image shape
        :param seg_dir:   for voc `SegmentationObject`,if seg_dir=None,will ignore Segmentation Object
        :param image_id: 初始的image_id,if None,will reset to currrent time
        """
        # init COCO Dataset struct
        self.coco = {"images": [], "annotations": [], "categories": [], "type": "instances"}
        self.category_set = dict()  # 类别集合
        self.image_set = set()  # 图片文件集合
        self.category_item_id = 0
        if not init_id: init_id = int(time.time()) * 2
        self.image_id = init_id
        # self.image_id = 20200207
        self.annotation_id = 0

    def save_coco(self, json_file):
        file_utils.create_file_path(json_file)
        file_utils.write_json_path(json_file, self.coco)
        print("save file:{}".format(json_file))

    def set_keypoints_category(self, kps_name=[], skeleton=[], cat_id=0):
        """
        设置关键点的名称和skeleton
        :param kps_name: 关键点的名称
        :param skeleton: 关键点连接点
        :param cat_id:
        :return:
        """
        # skeleton下标从0开始，coco_skeleton下标是从1开始的
        # skeleton = np.array(skeleton, dtype=np.int32) + 1
        assert self.coco['categories'], Exception("categories is empty")
        self.coco['categories'][cat_id]['keypoints'] = kps_name
        self.coco['categories'][cat_id]['skeleton'] = skeleton

    def addCatItem(self, name):
        """
        给COCO中categories增加类别
        :param name:
        :return:
        """
        self.category_item_id += 1
        category_item = dict()
        category_item['supercategory'] = name
        category_item['id'] = self.category_item_id
        category_item['name'] = name
        keypoints = []
        skeleton = []
        if name == "coco_person":
            keypoints = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
                         'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
                         'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
                         'right_ankle']
            skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7],
                        [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4],
                        [3, 5], [4, 6], [5, 7]]
        elif name == "finger":
            keypoints = {"finger0": 0, "finger1": 1, "finger2": 2, "finger3": 3, "finger4": 4,
                         "finger5": 5, "finger6": 6, "finger7": 7, "finger8": 8, "finger9": 9}
            keypoints = list(keypoints.keys())
            skeleton = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 1), (10, 3), (10, 5), (10, 7),
                        (10, 9)]
        elif name == "finger_pen":
            keypoints = {"finger0": 0, "finger1": 1, "finger2": 2, "finger3": 3, "finger4": 4,
                         "finger5": 5, "finger6": 6, "finger7": 7, "finger8": 8, "finger9": 9,
                         "finger10": 10, "pen0": 11, "pen1": 12}
            keypoints = list(keypoints.keys())
            skeleton = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 1), (10, 3), (10, 5), (10, 7), (10, 9), (11, 12)]
        # keypoints，skeleton
        category_item['keypoints'] = keypoints
        category_item['skeleton'] = skeleton
        self.coco['categories'].append(category_item)
        self.category_set[name] = self.category_item_id
        return self.category_item_id

    def addImgItem(self, file_name, image_size):
        """
        给COCO中images增加数据
        :param file_name:
        :param image_size: [height, width]
        :return:
        """
        if file_name is None:
            raise Exception('Could not find filename tag in xml file.')
        self.image_id += 1
        image_item = dict()
        image_item['id'] = self.image_id
        image_item['file_name'] = file_name
        image_item['height'] = image_size[0]
        image_item['width'] = image_size[1]
        self.coco['images'].append(image_item)
        self.image_set.add(file_name)
        return self.image_id

    def addAnnoItem(self, image_id, category_id, rect, seg, area, keypoints=[]):
        """
        给COCO中annotations增加数据
        :param image_id: int
        :param category_id: iny
        :param rect:[x,y,w,h]
        :param seg: [[轮廓1]，[轮廓2],....]格式为：[[x,y,x,y,...,x,y],[x,y,x,y,...,x,y],...]
        :param area: float,可通过get_segment_info获得seg和area
        :param keypoints: 关键点
                keypoints = [0] * 17 * 3
                keypoints = np.asarray(keypoints)
                keypoints[0::3] = np.clip(keypoints[0::3], 1, width - 1)
                keypoints[1::3] = np.clip(keypoints[1::3], 1, height - 1)
        :return:
        """
        self.annotation_id += 1
        annotation_item = dict()
        annotation_item['segmentation'] = seg
        annotation_item['area'] = area
        annotation_item['iscrowd'] = 0
        annotation_item['ignore'] = 0
        annotation_item['image_id'] = image_id  #
        annotation_item['bbox'] = rect  # [x,y,w,h]
        annotation_item['category_id'] = category_id
        annotation_item['id'] = self.annotation_id
        # if len(keypoints) > 0:
        #     annotation_item['num_keypoints'] = int(len(keypoints) / 3)
        #     annotation_item['keypoints'] = keypoints
        annotation_item['num_keypoints'] = int(len(keypoints) / 3)
        annotation_item['keypoints'] = keypoints
        self.coco['annotations'].append(annotation_item)

    def get_keypoints_info(self, keypoints, width, height, num_joints):
        """
        keypoints=num_joints*3,x,y,visibility
        keypoints关节点的格式 : [x_1, y_1, v_1,...,x_k, y_k, v_k]
        其中x,y为Keypoint的坐标，v为可见标志
            v = 0 : 未标注点
            v = 1 : 标注了但是图像中不可见（例如遮挡）
            v = 2 : 标注了并图像可见
        实际预测时，不要求预测每个关节点的可见性
        :param keypoints: (num_point,2)
        :param num_joints: 关键点个数
        :param width: 图像宽度
        :param height: 图像长度
        :return:
        """
        if len(keypoints) == 0: return []
        if isinstance(keypoints, list): keypoints = np.asarray(keypoints)
        kps = np.zeros(shape=(num_joints, 3), dtype=np.int32) + 2
        kps[:, 0:2] = keypoints
        kps[:, 0] = np.clip(kps[:, 0], 0, width - 1)
        kps[:, 1] = np.clip(kps[:, 1], 0, height - 1)
        kps = kps.reshape(-1).tolist()
        return kps

    def get_segment_info(self, contours, min_nums=2, min_area=10):
        """
        正常情况下，一个实例只有一条轮廓
        :param contours: (nums,points_nums,2)，(轮廓个数,轮廓点数,2),points_nums<4将会剔除
        :return: segs，分割轮廓保存格式 ：
                       [[x1,y1 x2,y2,... xn,yn], # 实例轮廓1
                        [x1,y1 x2,y2,... xn,yn], # 实例轮廓2
                        ]
                 area轮廓面积
        """
        # 计算轮廓面积
        segs, area = [], 0
        for contour in contours:
            if len(contour) < min_nums: continue
            contour = np.asarray(contour, dtype=np.int32)
            s = abs(cv2.contourArea(contour, True))
            seg = contour.reshape(-1).tolist()
            segs.append(seg)
            area += s
        if area < min_area: segs, area = [], []
        return segs, area

    def addObjects(self, filename, objects: dict, width, height, num_joints, ignore_tiny=False):
        """
        :param filename: os.path.basename(filename)
        :param objects:
                        boxes  is [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
                        labels is [目标1,目标2,...]
                        contours is [目标1的轮廓,目标2的轮廓,...]
                        keypoints is [目标1的关键点,目标2的关键点,...],格式：
                        keypoints=num_joints*3,x,y,visibility
                        keypoints关节点的格式 : [x_1, y_1, v_1,...,x_k, y_k, v_k]
                        其中x,y为Keypoint的坐标，v为可见标志
                            v = 0 : 未标注点
                            v = 1 : 标注了但是图像中不可见（例如遮挡）
                            v = 2 : 标注了并图像可见
        实际预测时，不要求预测每个关节点的可见性
        :param width:
        :param height:
        :param num_joints:
        :param ignore_tiny: 是否去除segment很细小的目标
        :return:
        """
        boxes = objects.get("boxes", [])
        labels = objects.get("labels", [])
        contours = objects.get("contours", [])
        keypoints = objects.get("keypoints", [])
        if not boxes: return
        if not labels: return
        assert len(boxes) == len(labels)
        if filename not in self.image_set:
            current_image_id = self.addImgItem(filename, image_size=[height, width])
        else:
            raise Exception('file_name duplicated: {}'.format(filename))
        for i in range(len(boxes)):
            name = labels[i]
            bbox = boxes[i]
            segs = contours[i] if contours else []
            kpts = keypoints[i] if keypoints else []
            if isinstance(bbox, np.ndarray): bbox = bbox.tolist()
            xmin, ymin, xmax, ymax = bbox
            rect = [xmin, ymin, xmax - xmin, ymax - ymin]
            # get segmentation info
            if len(segs) == 0: segs = [[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]]
            if ignore_tiny:
                segs, area = self.get_segment_info(segs, min_nums=2, min_area=10)
            else:
                segs, area = self.get_segment_info(segs, min_nums=0, min_area=0)
            if len(segs) == 0: continue
            if name not in self.category_set:
                current_category_id = self.addCatItem(name)
            else:
                current_category_id = self.category_set[name]
            self.addAnnoItem(current_image_id, current_category_id, rect, segs, area, keypoints=kpts)

    def build_keypoints_dataset_example(self, save_file, num_joints, class_name, kps_name=[], skeleton=[], **kwargs):
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
        dataset = []
        for index in tqdm(dataset):
            image_file = ...
            height, width = ...
            labels, boxes, contours, keypoints = [], [], [], []
            filename = os.path.basename(image_file)
            objects = {"boxes": boxes, "labels": labels, "contours": contours, "keypoints": keypoints}
            self.addObjects(filename, objects, width, height, num_joints)
        # 设置关键点的名称和skeleton
        kps_name = [str(i) for i in list(range(num_joints))]
        self.set_keypoints_category(kps_name=kps_name, skeleton=skeleton, cat_id=0)
        COCOTools.check_coco(self.coco)
        self.save_coco(save_file)
