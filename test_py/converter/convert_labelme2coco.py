# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: python-learning-notes
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-02-12 18:28:16
# @url    :
# --------------------------------------------------------
"""

import argparse
import sys
import os

sys.path.append(os.getcwd())
import glob
import numpy as np
import json
import xmltodict
import cv2
import PIL.Image
import time
import copy as copy
from tqdm import tqdm
from pybaseutils.dataloader import parser_labelme
from pybaseutils import file_utils, image_utils, color_utils, base64_utils


class COCOTools(object):
    """COCO Tools"""

    @staticmethod
    def get_categories_id(categories):
        """
        get categories id dict
        :param categories:
        :return: dict:{name:id}
        """
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


class Labelme2Coco(labelme_dataset.LabelMeDataset):
    """Convert Pascal VOC Dataset to COCO dataset format"""

    def __init__(self, json_dir, image_dir=None, init_id=None):
        """
        :param json_dir:  for voc `Annotations`
        :param image_dir: for voc `JPEGImages`,if image_dir=None ,will ignore checking image shape
        :param seg_dir:   for voc `SegmentationObject`,if seg_dir=None,will ignore Segmentation Object
        :param image_id: 初始的image_id,if None,will reset to currrent time
        """
        print(json_dir)
        print(image_dir)
        super(Labelme2Coco, self).__init__(json_dir, image_dir, skeleton=[])
        self.coco = dict()
        self.coco['images'] = []
        self.coco['type'] = 'instances'
        self.coco['annotations'] = []
        self.coco['categories'] = []

        self.category_set = dict()
        self.image_set = set()

        self.category_item_id = 0
        if not init_id:
            init_id = int(time.time()) * 2
        self.image_id = init_id
        # self.image_id = 20200207
        self.annotation_id = 0

    def addCatItem(self, name):
        """
        :param name:
        :return:
        """
        self.category_item_id += 1
        category_item = dict()
        category_item['supercategory'] = name
        category_item['id'] = self.category_item_id
        category_item['name'] = name
        if name == "person":
            category_item['keypoints'] = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
                                          'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
                                          'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
                                          'right_ankle']
            category_item['skeleton'] = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7],
                                         [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4],
                                         [3, 5], [4, 6], [5, 7]]
        elif name == "finger":
            skeleton_name = {"finger0": 0, "finger1": 1, "finger2": 2, "finger3": 3, "finger4": 4,
                             "finger5": 5, "finger6": 6, "finger7": 7, "finger8": 8, "finger9": 9}
            category_item['keypoints'] = list(skeleton_name.keys())
            # category_item['skeleton'] = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
            category_item['skeleton'] = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 1), (10, 3), (10, 5), (10, 7),
                                         (10, 9)]
        elif name == "finger_pen":
            skeleton_name = {"finger0": 0, "finger1": 1, "finger2": 2, "finger3": 3, "finger4": 4,
                             "finger5": 5, "finger6": 6, "finger7": 7, "finger8": 8, "finger9": 9,
                             "finger10": 10, "pen0": 11, "pen1": 12}

            category_item['keypoints'] = list(skeleton_name.keys())
            category_item['skeleton'] = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 1), (10, 3), (10, 5), (10, 7),
                                         (10, 9), (11, 12)]
        self.coco['categories'].append(category_item)
        self.category_set[name] = self.category_item_id
        return self.category_item_id

    def addImgItem(self, file_name, image_size, image_id):
        """
        :param file_name:
        :param image_size: [height, width]
        :return:
        """
        if file_name is None:
            raise Exception('Could not find filename tag in xml file.')
        image_item = dict()
        image_item['id'] = image_id
        image_item['file_name'] = file_name
        image_item['height'] = image_size[0]
        image_item['width'] = image_size[1]
        self.coco['images'].append(image_item)
        self.image_set.add(file_name)
        return image_id

    def addAnnoItem(self, image_id, category_id, rect, seg, area, keypoints):
        """
        :param image_id:
        :param category_id:
        :param rect:[x,y,w,h]
        :param seg: [[x1,y1,x2,y2,...,xn,yn,]]
        :param area:
        :param keypoints:
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
        annotation_item['num_keypoints'] = int(len(keypoints) / 3)
        annotation_item['keypoints'] = keypoints
        self.coco['annotations'].append(annotation_item)

    def generate_dataset(self, class_dict={}, vis=True):
        """
        :return:
        """
        for image_path in tqdm(self.image_list):
            filename = os.path.basename(image_path)
            json_file = os.path.join(self.json_dir, filename.split(".")[0] + ".json")
            if not os.path.exists(json_file):
                print("not exist:{}".format(json_file))
                continue
            image = cv2.imread(image_path)
            annos_points = self.read_json_annos(json_file)
            bboxes, labels, points = self.parser_annos_points(image, annos_points, class_dict, vis=False)
            height, width = image.shape[:2]
            if filename in self.category_set: raise Exception('file_name duplicated')
            if filename not in self.image_set:
                self.image_id += 1
                self.addImgItem(filename, image_size=[height, width], image_id=self.image_id)
                # print('add filename {}'.format(filename))
            else:
                raise Exception('duplicated image_dict: {}'.format(filename))
            dst_segmentation = []
            dst_categories = []
            dst_boxes = []
            for bbox, name, point in zip(bboxes, labels, points):
                if class_dict and name in class_dict:
                    name = class_dict[name]
                if name not in self.category_set:
                    current_category_id = self.addCatItem(name)
                else:
                    current_category_id = self.category_set[name]
                # get segmentation info
                seg, bbox = self.get_segmentation(image, point)
                xmin, ymin, xmax, ymax = bbox
                rect = [xmin, ymin, xmax - xmin, ymax - ymin]
                area = (xmax - xmin) * (ymax - ymin)
                keypoints = []
                dst_segmentation.append(seg)
                dst_categories.append(current_category_id)
                dst_boxes.append(bbox)
                self.addAnnoItem(self.image_id, current_category_id, rect, seg, area, keypoints)
            if vis: self.show_result(image, dst_categories, dst_boxes, seg=dst_segmentation, keypoint=[])
        COCOTools.check_coco(self.coco)

    def show_result(self, image, labels, boxes, seg=[], keypoint=[]):
        h, w = image.shape[:2]
        mask = self.draw_mask(seg, labels, height=h, width=w)
        image = np.asarray(image, np.uint8)
        mask = np.asarray(mask, np.uint8)
        color_image, color_mask = color_utils.decode_color_image_mask(image, mask)
        color_image = image_utils.draw_image_bboxes_labels_text(color_image, boxes, labels)
        vis = image_utils.image_hstack([image, mask, color_image, color_mask])
        # image_utils.cv_show_image("image", image)
        image_utils.cv_show_image("vis", vis)

    def draw_mask(self, contours, labels, height, width):
        mask = image_utils.create_image(shape=(height, width), color=(0, 0, 0))
        for point, label in zip(contours, labels):
            label = int(label)
            point = np.array(point).reshape(-1, 2)
            mask = image_utils.draw_image_fillPoly(mask, [point], color=(label, label, label))
        return mask

    def get_segmentation(self, image, points):
        """
        :param filename:
        :param bbox:[xmin, ymin, xmax, ymax]
        :return:
        """
        boxes = image_utils.polygons2boxes([points])
        boxes = boxes.tolist()[0]
        seg = [np.asarray(points).reshape(-1).tolist()]
        return seg, boxes

    def get_coco(self):
        return self.coco

    def save_coco(self, json_file):
        file_utils.create_file_path(json_file)
        coco = base64_utils.array2base64(self.get_coco())
        file_utils.write_json_path(json_file, coco)
        print("save coco file:{}".format(json_file))


def demo_retails():
    COCO_NAME = {}
    skeleton = [[0, 1], [1, 3], [3, 2], [2, 0], [4, 5]]
    data_root = "/home/dm/nasdata/dataset-dmai/handwriting/font-library-512/loss-word/缺失的字/word-image"
    image_dir = data_root
    json_dir = data_root
    save_path = os.path.join(os.path.dirname(data_root), "test_coco.json")
    VOC2coco = Labelme2Coco(json_dir, image_dir=image_dir)
    VOC2coco.generate_dataset(class_dict=COCO_NAME)
    VOC2coco.save_coco(save_path)
    file_utils.create_file_path(save_path)
    # demo_coco_test(annFile=save_path,
    #                image_dir=image_dir,
    #                COCO_NAME=COCO_NAME,
    #                skeleton=skeleton)


def demo_coco_test(annFile, image_dir, COCO_NAME, skeleton):
    import coco_demo
    co = coco_demo.CocoKeypoints(annFile, image_dir, COCO_NAME=COCO_NAME, skeleton=skeleton)
    # co = COCO_Instances(annFile, image_dir, COCO_NAME=None)
    # 获得所有图像id
    imgIds = co.getImgIds()
    # test_imgIds = imgIds[0:10]
    test_imgIds = imgIds

    # 显示目标的bboxes
    # info_list = co.get_object_rects(test_imgIds, show=True)
    # 显示关键点
    info_list = co.get_object_keypoints(test_imgIds, show=True)
    # 显示实例分割
    # co.get_object_instance(test_imgIds, show=True)
    # 显示语义分割的mask
    # co.get_object_mask(test_imgIds, show=True)
    # label编码
    # info_list = co.encode_info(info_list)
    # label解码
    # info_list = co.decode_info(info_list)
    # print("nums:{}".format(len(info_list)))
    # 保存label等信息
    # coordinatesType = "YOLO"
    # co.save_text_dataset(save_root=label_out_dir, info_list=info_list, coordinatesType=coordinatesType)
    # co.save_train_val_text_dataset(info_list, coco_root, label_out_dir)
    # batch_text_dataset_test(label_out_dir, image_dir, classes=COCO_NAME, coordinatesType=coordinatesType)

    # save_root = os.path.join(coco_root, "COCO")
    # co.save_voc_dataset(image_dir, info_list, save_root)


if __name__ == '__main__':
    demo_retails()
