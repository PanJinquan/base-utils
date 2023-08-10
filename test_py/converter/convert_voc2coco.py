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
from modules.dataset_tool.voc_tools.custom_voc import CustomVoc
from tqdm import tqdm
from utils import file_utils


def save_json(data_coco, json_file):
    """
    save COCO data in json file
    :param json_file:
    :return:
    """
    # json.dump(self.coco, open(json_file, 'w'))
    json.dump(data_coco, open(json_file, 'w'), indent=4)  # indent=4 更加美观显示
    # dirname = os.path.dirname(json_file)
    # if not os.path.exists(dirname):
    #     os.makedirs(dirname)
    # with open(json_file, 'w') as f:
    #     json.dump(data_coco, f, indent=4)
    print("save file:{}".format(json_file))


def read_json(json_path):
    """
    读取数据
    :param json_path:
    :return:
    """
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    return json_data


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


class PascalVoc2Coco(CustomVoc):
    """Convert Pascal VOC Dataset to COCO dataset format"""

    def __init__(self, anno_dir, image_dir=None, seg_dir=None, init_id=None):
        """
        :param anno_dir:  for voc `Annotations`
        :param image_dir: for voc `JPEGImages`,if image_dir=None ,will ignore checking image shape
        :param seg_dir:   for voc `SegmentationObject`,if seg_dir=None,will ignore Segmentation Object
        :param image_id: 初始的image_id,if None,will reset to currrent time
        """
        print(anno_dir)
        print(image_dir)
        self.anno_dir = anno_dir
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.xml_list = self.get_xml_files(self.anno_dir)
        # self.stroke_segs = SegmentationObject()

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

    def addImgItem(self, file_name, image_size):
        """
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

    def addAnnoItem(self, image_id, category_id, rect, seg, area, keypoints):
        """
        :param image_id:
        :param category_id:
        :param rect:[x,y,w,h]
        :param seg:
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

    def generate_dataset(self, class_dict={}):
        """
        :return:
        """
        for xml_file in tqdm(self.xml_list):
            # convert XML to Json
            content = self.read_xml2json(xml_file)
            annotation = content["annotation"]
            # get image shape
            width = int(annotation["size"]["width"])
            height = int(annotation["size"]["height"])
            depth = int(annotation["size"]["depth"])

            filename = annotation["filename"]
            self.check_image(filename, shape=(height, width, depth))
            if filename in self.category_set:
                raise Exception('file_name duplicated')

            if filename not in self.image_set:
                image_size = [height, width]
                current_image_id = self.addImgItem(filename, image_size=image_size)
                # print('add filename {}'.format(filename))
            else:
                raise Exception('duplicated image_dict: {}'.format(filename))
            try:
                objects = annotation["object"]
            except Exception as e:
                # raise Exception("Error:{}".format(xml_file))
                print("Error:{}".format(xml_file))
                continue

            if not isinstance(objects, list):
                objects = [objects]
            for object in objects:
                name = object["name"]
                if class_dict and name in class_dict:
                    name = class_dict[name]
                if name not in self.category_set:
                    current_category_id = self.addCatItem(name)
                else:
                    current_category_id = self.category_set[name]
                xmin = float(object["bndbox"]["xmin"])
                xmax = float(object["bndbox"]["xmax"])
                ymin = float(object["bndbox"]["ymin"])
                ymax = float(object["bndbox"]["ymax"])
                xmin = min(max(0, int(xmin)), width)
                xmax = min(max(0, int(xmax)), width)
                ymin = min(max(0, int(ymin)), height)
                ymax = min(max(0, int(ymax)), height)
                rect = [xmin, ymin, xmax - xmin, ymax - ymin]
                bbox = [xmin, ymin, xmax, ymax]

                # get person keypoints ,if exist
                if 'keypoints' in object:
                    keypoints = object["keypoints"]
                    keypoints = [float(i) for i in keypoints.split(",")]
                else:
                    keypoints = [0] * 17 * 3
                keypoints = np.asarray(keypoints)
                keypoints[0::3] = np.clip(keypoints[0::3], 1, width - 1)
                keypoints[1::3] = np.clip(keypoints[1::3], 1, height - 1)
                keypoints = keypoints.tolist()
                # get segmentation info
                seg, area = self.get_segmentation_area(filename, bbox=bbox)
                self.addAnnoItem(current_image_id, current_category_id, rect, seg, area, keypoints)
        COCOTools.check_coco(self.coco)

    def get_coco(self):
        return self.coco

    def save_coco(self, json_file):
        file_utils.create_file_path(json_file)
        save_json(self.get_coco(), json_file)


parser = argparse.ArgumentParser(description="COCO Dataset")
parser.add_argument("-i", "--image_dir", help="path/to/image", type=str)
parser.add_argument("-a", "--json_dir", help="path/to/json_dir", type=str)
parser.add_argument("-seg_dir", "--seg_dir", help="path/to/VOC/SegmentationObject", default=None, type=str)
parser.add_argument("-s", "--save_path", help="out/to/save_json-file", type=str)
parser.add_argument("-id", "--init_id", help="init id", type=int, default=None)
args = parser.parse_args()


def demo_finger():
    COCO_NAME = ["finger"]
    skeleton = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 1),
                (10, 3), (10, 5), (10, 7), (10, 9), (11, 12)]
    # data_root="/home/dm/data3/dataset/finger_keypoint/finger/orientation/landscape/"
    # data_root = "/home/dm/data3/dataset/finger_keypoint/finger/edge_test/"
    # data_root = "/data3/panjinquan/dataset/finger_keypoint/finger/train/"
    data_root = "/home/dm/data3/dataset/finger_keypoint/finger/train/"
    # data_root = "/home/dm/data3/dataset/finger_keypoint/finger/val/"
    # data_root = "/home/dm/data3/dataset/finger_keypoint/finger/landscape/"
    # data_root = "/data3/panjinquan/dataset/finger_keypoint/finger1/val/"
    image_dir = data_root + "images"
    anno_dir = data_root + 'annotations/xml'  # 这是xml文所在的地址
    save_path = data_root + "annotations/train_finger_pen_coco.json"
    # save_path = data_root + "annotations/val_finger_pen_coco.json"
    VOC2coco = PascalVoc2Coco(anno_dir, image_dir=image_dir, seg_dir=seg_dir, init_id=init_id)
    VOC2coco.generate_dataset()
    VOC2coco.save_coco(save_path)
    # demo_coco_test(annFile=save_path,
    #                image_dir=image_dir,
    #                COCO_NAME=COCO_NAME,
    #                skeleton=skeleton)


def demo_person_pen():
    COCO_NAME = ["person_pen"]
    skeleton = [[0, 1], [1, 3], [3, 2], [2, 0], [4, 5]]
    data_root = "/home/dm/data3/dataset/person_pen/v4/"
    image_dir = data_root + "images"
    anno_dir = data_root + 'xml/train'  # 这是xml文所在的地址
    save_path = data_root + "annotations/train_person_pen_coco.json"
    # save_path = data_root + "annotations/val_person_pen_coco.json"
    VOC2coco = PascalVoc2Coco(anno_dir, image_dir=image_dir, seg_dir=seg_dir, init_id=init_id)
    VOC2coco.generate_dataset()
    VOC2coco.save_coco(save_path)
    file_utils.create_file_path(save_path)
    # demo_coco_test(annFile=save_path,
    #                image_dir=image_dir,
    #                COCO_NAME=COCO_NAME,
    #                skeleton=skeleton)


def demo_retails():
    COCO_NAME = {'Ping': "retails", 'Guan': "retails", 'He': "retails", 'Ling': "retails"}
    skeleton = [[0, 1], [1, 3], [3, 2], [2, 0], [4, 5]]
    data_root = "/home/dm/nasdata/dataset/new_retails/VOC/collect_ip_360_val/"
    image_dir = data_root + "JPEGImages"
    anno_dir = data_root + 'Annotations'  # 这是xml文所在的地址
    save_path = data_root + "annotations/train_person_pen_coco.json"
    # save_path = data_root + "annotations/val_person_pen_coco.json"
    VOC2coco = PascalVoc2Coco(anno_dir, image_dir=image_dir, seg_dir=seg_dir, init_id=init_id)
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
    image_dir = args.image_dir
    seg_dir = args.seg_dir
    save_path = args.save_path
    init_id = args.init_id
    # demo_person_pen()
    demo_retails()
    # demo_finger()
