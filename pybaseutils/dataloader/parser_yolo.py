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
import numbers
import torch
import json
from tqdm import tqdm
from pybaseutils import image_utils, file_utils, coords_utils
from pybaseutils.dataloader.base_dataset import Dataset


class YOLODataset(Dataset):
    """
    YOLO数据格式解析器
    数据格式：(class,cx,cy,w,h)/(1,width,height,width,height)，将归一化的Text数据
    输出格式：box is (xmin,ymin,xmax,ymax)
    """

    def __init__(self,
                 filename=None,
                 data_root=None,
                 anno_dir=None,
                 image_dir=None,
                 class_name=None,
                 use_rgb=True,
                 shuffle=False,
                 check=False,
                 **kwargs):
        """
        Each row is [class cx cy w h](class x_center y_center width height) format;
        Box coordinates must be in normalized xywh format (from 0 - 1).
        If your boxes are in pixels, divide x_center and width by image width, and y_center and height by image height.
        such as:[0, 0.9146634615384616, 0.3497596153846154, 0.11298076923076923, 0.14182692307692307]
        ----------------------------------------
        .
        ├── images
        │       ├── 0001.jpg
        │       ├── 0002.jpg
        │       ├── 0003.jpg
        │       └── 0004.jpg
        ├── labels
        │       ├── 0001.txt
        │       ├── 0002.txt
        │       ├── 0003.txt
        │       └── 0004.txt
        └── train.txt
        ----------------------------------------
        :param filename:
        :param data_root:
        :param anno_dir:
        :param image_dir:
        :param transform:
        :param use_rgb:
        :param shuffle:
        """
        super(YOLODataset, self).__init__()
        self.min_area = 1 / 1000  # 如果前景面积不足0.1%,则去除
        self.use_rgb = use_rgb
        self.class_name, self.class_dict = self.parser_classes(class_name)
        parser = self.parser_paths(filename, data_root, anno_dir, image_dir)
        self.data_root, self.anno_dir, self.image_dir, self.image_ids = parser
        self.image_ids = self.add_image_postfix(self.image_dir, self.image_ids)
        self.classes = list(self.class_dict.values()) if self.class_dict else None
        self.num_classes = max(list(self.class_dict.values())) + 1 if self.class_dict else None
        self.class_weights = None
        if check:
            self.image_ids = self.checking(self.image_ids)
        if shuffle:
            random.seed(200)
            random.shuffle(self.image_ids)
        self.num_images = len(self.image_ids)
        self.scale_rate = 1.0
        self.target_type = 'gaussian'
        self.sigma = 2
        print("Dataset class_name    :{}".format(class_name))
        print("Dataset class_dict    :{}".format(self.class_dict))
        print("Dataset num images    :{}".format(len(self.image_ids)))
        print("Dataset num_classes   :{}".format(self.num_classes))
        print("------" * 10)

    def __len__(self):
        return len(self.image_ids)

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
            class_name = super().read_file(class_name)
        elif isinstance(class_name, numbers.Number):
            class_name = [str(i) for i in range(int(class_name))]
        elif isinstance(class_name, list) and "unique" in class_name:
            self.unique = True
        if isinstance(class_name, list) and len(class_name) > 0:
            class_dict = {str(class_name): i for i, class_name in enumerate(class_name)}
        elif isinstance(class_name, dict) and len(class_name) > 0:
            class_dict = class_name
            class_name = list(class_dict.keys())
        else:
            class_dict = None
        return class_name, class_dict

    def add_image_postfix(self, image_dir, image_ids):
        """
        获得图像文件后缀名
        :param image_dir:
        :return:
        """
        if "." not in image_ids[0]:
            image_list = glob.glob(os.path.join(image_dir, "*"))
            postfix = os.path.basename(image_list[0]).split(".")[1]
            image_ids = [f"{image_id}.{postfix}" for image_id in image_ids]
        return image_ids

    def get_image_anno_file(self, index):
        """
        :param index:
        :return:
        """
        image_id = self.index2id(index)
        image_file, annotation_file, image_id = self.__get_image_anno_file(self.image_dir,
                                                                           self.anno_dir,
                                                                           image_id)
        return image_file, annotation_file, image_id

    def __get_image_anno_file(self, image_dir, anno_dir, image_name: str):
        """
        :param image_dir:
        :param anno_dir:
        :param image_id:
        :return:
        """
        img_postfix = image_name.split(".")[-1]
        image_id = image_name[:-len(img_postfix) - 1]
        image_file = os.path.join(image_dir, "{}.{}".format(image_id, img_postfix))
        annotation_file = os.path.join(anno_dir, "{}.txt".format(image_id))
        return image_file, annotation_file, image_id

    def checking(self, image_ids: list, ignore_empty=True):
        """
        :param image_ids:
        :param ignore_empty : 是否去除一些空数据
        :return:
        """
        print("Please wait, it's in checking")
        dst_ids = []
        # image_ids = image_ids[:100]
        # image_ids = image_ids[100:]
        for image_id in tqdm(image_ids):
            image_file, annotation_file, image_id = self.get_image_anno_file(image_id)
            if not os.path.exists(annotation_file):
                continue
            if not os.path.exists(image_file):
                continue
            annotation = self.load_annotations(annotation_file)
            if len(annotation) == 0:
                continue
            dst_ids.append(image_id)
        print("have nums image:{},legal image:{}".format(len(image_ids), len(dst_ids)))
        return dst_ids

    def parser_paths(self, filename=None, data_root=None, anno_dir=None, image_dir=None):
        """
        :param filename:
        :param data_root:
        :param anno_dir:
        :param image_dir:
        :return:
        """
        if isinstance(data_root, str):
            anno_dir = os.path.join(data_root, "json") if not anno_dir else anno_dir
            image_dir = os.path.join(data_root, "images") if not image_dir else image_dir
        image_ids = []
        if isinstance(filename, str) and filename:
            image_ids = self.read_file(filename, split=",")
            data_root = os.path.dirname(filename)
        if not anno_dir:  # 如果anno_dir为空，则自动搜寻可能存在图片目录
            image_sub = ["labels"]
            anno_dir = self.search_path(data_root, image_sub)
        if not image_dir:
            image_dir = self.search_path(data_root, ["JPEGImages", "images"])
        if anno_dir and not image_ids:
            image_ids = self.get_file_list(anno_dir, postfix=["*.json"], basename=False)
            image_ids = [os.path.basename(f) for f in image_ids]
        elif image_dir and not image_ids:
            image_ids = self.get_file_list(anno_dir, postfix=["*.jpg"], basename=False)
            image_ids = [os.path.basename(f) for f in image_ids]
        # assert os.path.exists(image_dir), Exception("no directory:{}".format(image_dir))
        # assert os.path.exists(anno_dir), Exception("no directory:{}".format(anno_dir))
        assert len(image_ids) > 0, f"image_dir={image_dir} is empty"
        return data_root, anno_dir, image_dir, image_ids

    def __getitem__(self, index):
        """
        :param index: int or str
        :return:rgb_image
        """
        image_id = self.index2id(index)
        image_file, anno_file, image_id = self.get_image_anno_file(image_id)
        image = self.read_image(image_file, use_rgb=self.use_rgb)
        shape = image.shape
        annotation = self.load_annotations(anno_file)
        boxes, labels, points = self.parser_annotation_segs(annotation, shape, self.class_dict)
        data = {"image": image, "boxes": boxes, "labels": labels, "points": points,
                "image_file": image_file, "anno_file": anno_file}
        return data

    @staticmethod
    def parser_annotation_boxes(annotation: dict, shape, class_dict):
        """
        :param annotation:  labelme标注的数据
        :param class_dict:  label映射
        :param shape: 图片shape(H,W,C),可进行坐标点的维度检查，避免越界
        :return:
        """
        # dim=5,annotation is [class_index, cx, cy, w,  h]
        # dim=9,annotation is [class_index, x1, y1, x2, y2, x3, y3, x4, y4],四个角点
        annotation = np.asarray(annotation)
        num, dim = annotation.shape
        labels = annotation[:, 0:1].astype(int)
        points = annotation[:, 1:]
        h, w = shape[:2]
        if dim == 5:
            if shape:
                bboxes = coords_utils.cxcywh2xyxy(points, width=w, height=h, normalized=True)
            else:
                bboxes = coords_utils.cxcywh2xyxy(points)
            points = image_utils.boxes2polygons(bboxes)
        else:
            points = points.reshape(-1, 4, 2)
            points = points * [w, h]
            bboxes = image_utils.points2bbox(points)
        return bboxes, labels, points

    @staticmethod
    def parser_annotation_segs(annotation: dict, shape, class_dict):
        """
        :param annotation:  labelme标注的数据
        :param class_dict:  label映射
        :param shape: 图片shape(H,W,C),可进行坐标点的维度检查，避免越界
        :return:
        """
        # dim=5,annotation is [class_index, cx, cy, w,  h]
        # dim=9,annotation is [class_index, x1, y1, x2, y2, x3, y3, x4, y4],四个角点
        h, w = shape[:2]
        bboxes, labels, points = [], [], []
        for anno in annotation:
            label = anno[0]
            polys = np.asarray(anno[1:]).reshape(-1, 2)
            polys = polys * [w, h]
            boxes = image_utils.polygons2boxes([polys])[0]
            labels.append(label)
            points.append(polys)
            bboxes.append(boxes)
        return bboxes, labels, points

    def index2id(self, index):
        """
        :param index: int or str
        :return:
        """
        if isinstance(index, numbers.Number):
            image_id = self.image_ids[index]
        else:
            image_id = index
        return image_id

    def __len__(self):
        return len(self.image_ids)

    @staticmethod
    def get_files_id(file_list):
        """
        :param file_list:
        :return:
        """
        image_idx = []
        for path in file_list:
            basename = os.path.basename(path)
            id = basename.split(".")[0]
            image_idx.append(id)
        return image_idx

    def read_image(self, image_file: str, use_rgb=True):
        """
        :param image_file:
        :param use_rgb:
        :return:
        """
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = image[:, :, 0:3]
        if use_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @staticmethod
    def load_annotations(ann_file: str):
        annos = file_utils.read_data(ann_file, split=" ")
        return annos


def parser_labelme(anno_file, class_dict={}, shape=None):
    """
    :param annotation:  labelme标注的数据
    :param class_dict:  label映射
    :param shape: 图片shape(H,W,C),可进行坐标点的维度检查，避免越界
    :return:
    """
    annotation = YOLODataset.load_annotations(anno_file)
    bboxes, labels, points = YOLODataset.parser_annotation(annotation, shape, class_dict)
    return bboxes, labels


def show_target_image(image, bboxes, labels, points=[], class_name=None, use_rgb=True):
    """
    :param image:
    :param bboxes:
    :param labels:
    :param points: [shape(n,2),[shape(n,2)]]
    :param class_name:
    :param use_rgb:
    :return:
    """
    dst = image.copy()
    dst = image_utils.draw_image_contours(dst, contours=points, alpha=0.1, thickness=1)
    dst = image_utils.image_hstack([image, dst])
    image_utils.cv_show_image("det", dst, use_rgb=use_rgb)


if __name__ == "__main__":
    # filename = "/home/dm/nasdata/dataset/csdn/helmet/helmet-dataset-v2/train.txt"
    # filename = "/home/dm/nasdata/dataset/csdn/helmet/helmet-asian/total.txt"
    # filename = "/home/dm/nasdata/dataset/csdn/helmet/helmet-asian/total.txt"
    filename = "/home/PKing/nasdata/tmp/tmp/medical/肾结石CT/valid/sample.txt"
    dataset = YOLODataset(filename=filename,
                          data_root=None,
                          anno_dir=None,
                          image_dir=None,
                          class_name=None,
                          check=False,
                          phase="val",
                          shuffle=False)
    print("have num:{}".format(len(dataset)))
    for i in range(len(dataset)):
        print(i)  # i=20
        data = dataset.__getitem__(i)
        image, bboxes, labels = data["image"], data["boxes"], data["labels"]
        points = data["points"]
        h, w = image.shape[:2]
        image_file = data["image_file"]
        show_target_image(image, bboxes, labels, points=points)
