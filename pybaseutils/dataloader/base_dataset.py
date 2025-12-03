# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2025-04-29 09:13:09
# @Brief  :
# --------------------------------------------------------
"""
import os
import numpy as np
import cv2
import glob
import random
import xmltodict
import json
from pybaseutils import file_utils, json_utils, text_utils, image_utils

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
        self.tag = self.__class__.__name__
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
            # class_dict = json_utils.dict_sort(class_dict, reverse=False)
            class_name = {}
            for n, i in class_dict.items():
                class_name[i] = "{},{}".format(class_name[i], n) if i in class_name else n
            class_name = list(class_name.values())
        if isinstance(class_name, list) and "unique" in class_name:
            self.unique = True
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
    def get_file_list(dir_root, postfix=['*.jpg'], sub=False, basename=True):
        """
        获得文件列表
        :param dir_root: 图片文件目录
        :param postfix: 后缀名，可是多个如，['*.jpg','*.png']
        :param sub: 是否去除根路径
        :param basename: 返回的列表是文件名（True），还是文件的完整路径(False)
        :return:
        """
        file_list = file_utils.get_files_list(dir_root, prefix="", postfix=postfix, sub=sub, basename=basename)
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

    def __init__(self, datasets, shuffle=False, **kwargs):
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
        self.log = kwargs.get('log', print) if kwargs.get('log', print) else print
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
        self.log("{:15s} total images  :{}".format(self.tag, len(self.image_ids)))
        self.log("{:15s} class_name    :{}".format(self.tag, self.class_name))
        self.log("------" * 10)

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


def count_class_info(item_list, class_name=None, label_index="label"):
    """
    统计类别信息
    item_list=[[file,label,...],[file,label,...]]
    :param item_list:
    :param class_name:
    :return:
    """
    count = {}
    for item in item_list:
        label = item[label_index]
        count[label] = count[label] + 1 if label in count else 1
    count = json_utils.dict_sort(count, use_key=True)
    if class_name: count = {class_name[k]: v for k, v in count.items()}
    return count


def get_targets_index(obj_info: dict, index, nums, keys=[], out_info={}):
    """从obj_info中获得第index个数据，并拼接在out_info"""
    if not obj_info: return out_info
    if keys and out_info: [out_info.pop(k) for k in list(out_info.keys()) if k not in keys]
    for k, v in obj_info.items():
        if keys and k not in keys: continue  # 如果指定了keys,则进行过滤
        if isinstance(v, np.ndarray): v = v.tolist()
        if isinstance(v, list) and nums == len(v):
            out_info[k] = out_info.get(k, []) + [v[index]]
        else:
            out_info[k] = v
    return out_info


def cat_targets(objs: list, keys=[], out_info={}):
    """
    将相同字段的数据进行拼接
    :param objs: List(dict)
    :param keys:
    :param out_info:
    :return:
    """
    if not objs: return out_info
    if keys and out_info: [out_info.pop(k) for k in list(out_info.keys()) if k not in keys]
    for obj in objs:
        for k, v in obj.items():
            if keys and k not in keys: continue  # 如果指定了keys,则进行过滤
            if isinstance(v, np.ndarray): v = v.tolist()
            if isinstance(v, list):
                out_info[k] = out_info.get(k, []) + obj.get(k, [])
            else:
                out_info[k] = out_info.get(k, []) + [v]
    return out_info


def get_targets(obj_info: dict, targets=[], key='names', keys=[]):
    """
    从obj_info查找符合条件的目标，支持正则表达式
    :param obj_info:
    :param targets: 选择过滤的目标
    :param key: 选择targets对于key字段
    :param keys: 用于指定返回的keys
    :return:
    """
    output = {k: [] for k in keys}
    if not obj_info: return output
    label = obj_info[key]
    for i in range(len(label)):
        matches = text_utils.find_match_texts(texts=[label[i]], pattern=targets, org=True)
        if len(matches) == 0: continue
        get_targets_index(obj_info, index=i, nums=len(label), keys=keys, out_info=output)
    return output


def get_targets_overlap(obj_info1: dict, obj_info2: dict, key="boxes", keys=[], iou_th=0, use_iom=False):
    """
    :param obj_info1: dict(boxes=目标框(xmin,ymin,xmax,ymax),labels=类别名称),
    :param obj_info2: dict(boxes=目标框(xmin,ymin,xmax,ymax),labels=类别名称)
    :param key: 选择targets对于key字段
    :param keys: 用于指定返回的keys
    :param iou_th: 返回IOU>iou_th的object(不含等于)
    :param use_iom: IOU=交集(A,B)/并集(A,B),IOM=交集(A,B)/最小集(A,B)
    :return:output 返回列表，每个元素格式
                 {
                      'boxes': [[10, 10, 50, 50]], # obj1查询目标框
                      'index': 0,                  # obj1目标index
                      'label': ['A0'],             # obj1目标label
                      'maxiou': 1,                 # 在match中，obj1与obj2最大IOU值的下标
                      'match': [                   # obj2目标信息
                                {
                                  'boxes': [[20, 20, 40, 60]],
                                  'index': 0,
                                  'iou': 0.3333,
                                  'label': ['B0']
                                },
                                {
                                  'boxes': [[20, 20, 45, 60]],
                                  'index': 2,
                                  'iou': 0.4054,
                                  'label': ['B2']
                                }
                              ]
                 }
    """
    if not obj_info1: return []
    if not obj_info2: return []
    boxes1 = obj_info1[key]
    boxes2 = obj_info2[key]
    nums1 = len(boxes1)
    nums2 = len(boxes2)
    ious = image_utils.get_boxes_iom(boxes1, boxes2) if use_iom else image_utils.get_boxes_iou(boxes1, boxes2)
    output = []
    for i in range(len(boxes1)):
        outs = get_targets_index(obj_info1, index=i, nums=nums1, keys=keys, out_info={})
        obj2 = []
        for j in range(len(boxes2)):
            iou = ious[i, j]
            if iou > iou_th:
                item = get_targets_index(obj_info2, index=j, nums=nums2, keys=keys, out_info={})
                item.update(iou=iou, index=j)
                obj2.append(item)
        maxiou = int(np.argmax([data['iou'] for data in obj2])) if obj2 else -1
        outs.update(index=i, maxiou=maxiou, match=obj2)
        output.append(outs)
    return output


if __name__ == '__main__':
    #
    obj_info1 = {'boxes': [[10, 10, 50, 50],
                           [10, 10, 50, 50]],
                 "label": ["A0", "A1"],
                 "info1": {}}
    obj_info2 = {'boxes': [[20, 20, 40, 60],
                           [60, 60, 80, 80],
                           [20, 20, 45, 60],
                           ],
                 "label": ["B0", "B1", "B2"],
                 "info2": {}}
    obj_info2 = get_targets(obj_info2, targets=['B2', 'B0'], key='label', keys=['boxes'])
    print(json_utils.formatting(obj_info2))
    print("--------" * 10)
    output = get_targets_overlap(obj_info1, obj_info2, iou_th=-1)
    match = output[0]['match']
    match = cat_targets(match)
    print(json_utils.formatting(output))
