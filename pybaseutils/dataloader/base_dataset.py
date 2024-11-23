# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail :
    @Date   : 2023-08-10 10:18:32
    @Brief  :
"""
import os
import numpy as np
import cv2
import glob
import random
import xmltodict
import json
from pybaseutils import file_utils, json_utils

VOC_NAMES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
             "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
             "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

COCO_NAMES = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
              'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
              'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
              'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
              'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
              'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
              'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
              'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
              'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
              'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
              'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
              'teddy bear', 'hair drier', 'toothbrush']


class Dataset(object):
    """
    from torch.utils.data import Dataset,DataLoader, ConcatDataset
    """

    def __init__(self, **kwargs):
        self.image_ids = []
        # TODO: self.class_name, self.class_dict = self.parser_classes(class_name)
        self.class_name = []
        self.class_dict = []
        # TODO: self.classes = list(self.class_dict.values()) if self.class_dict else None
        self.classes = []
        self.postfix = "jpg"
        self.unique = False  # 是否是单一label，如["BACKGROUND", "unique"]

    def __getitem__(self, index):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])

    def __len__(self):
        raise NotImplementedError

    def parser_classes(self, class_name):
        """
        class_dict = {class_name: i for i, class_name in enumerate(class_name)}
        :param class_name:
                    str : class file
                    list: ["face","person"]
                    dict: 可以自定义label的id{'BACKGROUND': 0, 'person': 1, 'person_up': 1, 'person_down': 1}
        :return:
        """
        if isinstance(class_name, str):
            class_name = Dataset.read_file(class_name)
        elif isinstance(class_name, list) and "unique" in class_name:
            self.unique = True
        if isinstance(class_name, list) and len(class_name) > 0:
            class_dict = {}
            for i, name in enumerate(class_name):
                name = name.split(",")
                for n in name: class_dict[n] = i
        elif isinstance(class_name, dict) and len(class_name) > 0:
            class_dict = class_name
            class_name = list(class_dict.keys())
        else:
            class_dict = None
        if class_dict:
            class_dict = json_utils.dict_sort_by_value(class_dict, reverse=False)
            class_name = {}
            for n, i in class_dict.items():
                class_name[i] = "{},{}".format(class_name[i], n) if i in class_name else n
            class_name = list(class_name.values())
        return class_name, class_dict

    @staticmethod
    def read_xml2json(file):
        """
        import xmltodict
        :param file:
        :return:
        """
        with open(file, encoding='utf-8') as fd:  # 将XML文件装载到dict里面
            content = xmltodict.parse(fd.read())
        return content

    @staticmethod
    def read_json_data(file):
        """
        读取数据
        :param file:
        :return:
        """
        with open(file, 'rb') as f:
            json_data = json.load(f)
        return json_data

    @staticmethod
    def read_file(filename, split=None):
        """
        :param filename:
        :param split:分割
        :return:
        """
        image_id = []
        with open(filename, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip().split(split) if split else line.rstrip()
                line = line[0] if isinstance(line, list) and len(line) == 1 else line
                image_id.append(line)
        return image_id

    @staticmethod
    def read_files(filename, split=None):  # TODO 避免兼容问题
        return Dataset.read_file(filename, split=split)

    @staticmethod
    def get_file_list(dir_root, postfix=['*.jpg'], basename=True):
        """
        获得文件列表
        :param dir_root: 图片文件目录
        :param postfix: 后缀名，可是多个如，['*.jpg','*.png']
        :param basename: 返回的列表是文件名（True），还是文件的完整路径(False)
        :return:
        """
        file_list = []
        for format in postfix:
            format = os.path.join(dir_root, format)
            image_list = glob.glob(format)
            if image_list:
                file_list += image_list
        file_list = sorted(file_list)
        if basename:
            file_list = [os.path.basename(f).split(".")[0] for f in file_list]
        return file_list

    @staticmethod
    def xyxy2cxcywh(xyxy, width=None, height=None, norm=False):
        """
        将(xmin, ymin, xmax, ymax)转换为(cx,cy,w,h)
        """
        cxcywh = np.zeros_like(xyxy)
        cxcywh[:, 0] = (xyxy[:, 2] + xyxy[:, 0]) / 2  # cx
        cxcywh[:, 1] = (xyxy[:, 3] + xyxy[:, 1]) / 2  # cy
        cxcywh[:, 2] = (xyxy[:, 2] - xyxy[:, 0])  # w
        cxcywh[:, 3] = (xyxy[:, 3] - xyxy[:, 1])  # h
        # xyxy = np.concatenate([cxcywh[..., :2] - cxcywh[..., 2:] / 2,
        #                        cxcywh[..., :2] + cxcywh[..., 2:] / 2], axis=1)
        if norm:
            cxcywh = cxcywh / (width, height, width, height)
        return cxcywh

    @staticmethod
    def cxcywh2xyxy(cxcywh, width=None, height=None, unnorm=False):
        """
        将(cx,cy,w,h)转换为(xmin, ymin, xmax, ymax)
        """
        xyxy = np.zeros_like(cxcywh)
        xyxy[:, 0] = cxcywh[:, 0] - cxcywh[:, 2] / 2  # top left x
        xyxy[:, 1] = cxcywh[:, 1] - cxcywh[:, 3] / 2  # top left y
        xyxy[:, 2] = cxcywh[:, 0] + cxcywh[:, 2] / 2  # bottom right x
        xyxy[:, 3] = cxcywh[:, 1] + cxcywh[:, 3] / 2  # bottom right y
        if unnorm:
            xyxy = xyxy * (width, height, width, height)
        return xyxy

    @staticmethod
    def clip_box(box, width, height):
        # xmin, ymin, xmax, ymax = bbox
        box[0] = max(0, box[0])
        box[1] = max(0, box[1])
        box[2] = min(width - 1, box[2])
        box[3] = min(height - 1, box[3])
        return box

    @staticmethod
    def check_box(box, width, height):
        xmin, ymin, xmax, ymax = box
        sw = (xmax - xmin) / width
        sh = (ymax - ymin) / height
        ok = True
        if sw < 0 or sw > 1:
            ok = False
        elif sh < 0 or sh > 1:
            ok = False
        return ok

    @staticmethod
    def search_path(root, sub_dir=[]):
        """搜索可能存在的子目录"""
        for sub in sub_dir:
            path = os.path.join(root, sub)
            if os.path.exists(path):
                return path
        return None


class ConcatDataset(Dataset):
    """ Concat Dataset """

    def __init__(self, datasets, shuffle=False):
        """
        import torch.utils.data as torch_utils
        voc1 = PolygonParser(filename1)
        voc2 = PolygonParser(filename2)
        voc=torch_utils.ConcatDataset([voc1, voc2])
        ====================================
        :param datasets:
        :param shuffle:
        """
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'dataset should not be an empty iterable'
        # super(ConcatDataset, self).__init__()
        if not isinstance(datasets, list):
            datasets = [datasets]
        self.image_ids = []
        self.dataset = datasets
        self.shuffle = shuffle
        for dataset_id, dataset in enumerate(self.dataset):
            image_ids = dataset.image_ids
            image_ids = self.add_dataset_id(image_ids, dataset_id)
            self.image_ids += image_ids
            self.classes = dataset.classes
            self.class_name = dataset.class_name
        if shuffle:
            random.seed(200)
            random.shuffle(self.image_ids)
        print("ConcatDataset total images :{}".format(len(self.image_ids)))
        print("ConcatDataset class_name   :{}".format(self.class_name))
        print("------" * 10)

    def add_dataset_id(self, image_ids, dataset_id):
        """
        :param image_ids:
        :param dataset_id:
        :return:
        """
        out_image_id = []
        for image_id in image_ids:
            out_image_id.append({"dataset_id": dataset_id, "image_id": image_id})
        return out_image_id

    def __getitem__(self, index):
        """
        :param index: int
        :return:
        """
        dataset_id = self.image_ids[index]["dataset_id"]
        image_id = self.image_ids[index]["image_id"]
        dataset = self.dataset[dataset_id]
        data = dataset.__getitem__(image_id)
        return data

    def get_image_anno_file(self, index):
        dataset_id = self.image_ids[index]["dataset_id"]
        image_id = self.image_ids[index]["image_id"]
        return self.dataset[dataset_id].get_image_anno_file(image_id)

    def get_annotation(self, anno_file):
        return self.dataset[0].get_annotation(anno_file)

    def read_image(self, image_file):
        return self.dataset[0].read_image(image_file, use_rgb=self.dataset[0].use_rgb)

    def __len__(self):
        return len(self.image_ids)
