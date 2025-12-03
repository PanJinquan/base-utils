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
                 task="",
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
        :param class_name:
        :param task:  任务类型:det,obb,seg,pose:
        :param use_rgb:
        :param shuffle:
        """
        self.tag = self.__class__.__name__
        super(YOLODataset, self).__init__()
        self.log = kwargs.get('log', print) if kwargs.get('log', print) else print
        self.min_area = 1 / 1000  # 如果前景面积不足0.1%,则去除
        self.use_rgb = use_rgb
        self.task = task
        self.class_name, self.class_dict = self.parser_classes(class_name)
        parser = self.parser_paths(filename, data_root, anno_dir, image_dir)
        self.data_root, self.anno_dir, self.image_dir, self.image_ids = parser
        self.image_ids = self.parser_dataset(self.image_dir, self.image_ids)
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
        self.log("{:15s} class_name    :{}".format(self.tag, class_name))
        self.log("{:15s} class_dict    :{}".format(self.tag, self.class_dict))
        self.log("{:15s} num images    :{}".format(self.tag, len(self.image_ids)))
        self.log("{:15s} num_classes   :{}".format(self.tag, self.num_classes))
        self.log("------" * 10)

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

    def parser_dataset(self, image_dir, image_ids):
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
        image_file, anno_file, image_id = self.__get_image_anno_file(self.image_dir,  self.anno_dir, image_id)
        return image_file, anno_file, image_id

    def __get_image_anno_file(self, image_dir, anno_dir, image_name: str):
        """
        :param image_dir:
        :param anno_dir:
        :param image_id:
        :return:
        """
        image_id, img_postfix = file_utils.split_postfix(image_name)
        image_file = os.path.join(image_dir, "{}.{}".format(image_id, img_postfix))
        anno_file = os.path.join(anno_dir, "{}.txt".format(image_id))
        return image_file, anno_file, image_id

    def checking(self, image_ids: list, ignore_empty=True):
        """
        :param image_ids:
        :param ignore_empty : 是否去除一些空数据
        :return:
        """
        dst_ids = []
        for image_id in tqdm(image_ids, desc="check data"):
            image_file, anno_file, image_id = self.get_image_anno_file(image_id)
            if not os.path.exists(anno_file):
                continue
            if not os.path.exists(image_file):
                continue
            annotation = self.load_annotations(anno_file)
            if len(annotation) == 0:
                continue
            dst_ids.append(os.path.basename(image_file))
        self.log("have nums image:{},legal image:{}".format(len(image_ids), len(dst_ids)))
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
            anno_dir = os.path.join(data_root, "labels") if not anno_dir else anno_dir
            image_dir = os.path.join(data_root, "images") if not image_dir else image_dir
        image_ids = []
        if isinstance(filename, str) and filename:
            image_ids = self.read_file(filename, split=",")
            data_root = os.path.dirname(filename)
        if not anno_dir:  # 如果anno_dir为空，则自动搜寻可能存在图片目录
            anno_dir = self.search_path(data_root, sub_dir=["labels"])
        if not data_root and anno_dir:  #
            data_root = os.path.dirname(anno_dir)
            image_dir = self.search_path(data_root, ["images", "JPEGImages"])
        if not image_dir:
            image_dir = self.search_path(data_root, ["images", "JPEGImages"])
        if image_dir and not image_ids:
            image_ids = self.get_file_list(image_dir, postfix=file_utils.IMG_POSTFIX, sub=True, basename=False)
            if not anno_dir: anno_dir = image_dir
        elif anno_dir and not image_ids:
            image_ids = self.get_file_list(anno_dir, postfix=file_utils.IMG_POSTFIX, sub=True, basename=False)
            if not image_dir: image_dir = anno_dir
        assert isinstance(anno_dir, str) and os.path.exists(anno_dir), "no anno_dir :{}".format(anno_dir)
        assert isinstance(image_dir, str) and os.path.exists(image_dir), "no image_dir:{}".format(image_dir)
        assert len(image_ids) > 0, f"image_ids is empty,image_dir={image_dir},anno_dir={anno_dir}"
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
        boxes, labels, points = self.parser_annotation(annotation, shape, task=self.task)
        names = [self.class_name[i] for i in labels] if self.class_name else labels
        data = {"image": image, "boxes": boxes, "labels": labels, "names": names, "points": points,
                "image_file": image_file, "anno_file": anno_file}
        return data

    @staticmethod
    def parser_annotation(annotation: dict, shape, task=""):
        """
        - 检测任务  size=5   <class-index> <x_center> <y_center> <width> <height>
        - 分割任务: size=n   <class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>
        - OBB任务: size=9   <class-index> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
        - 姿态估计: size=5+n <class-index> <x_center> <y_center> <width> <height> <px1> <py1> <px2> <py2> ... <pxn> <pyn>
                        或者<class-index> <x_center> <y_center> <width> <height> <px1> <py1> <p1-vis> <px2> <py2> <p2-vis> <pxn> <pyn> <pn-vis>
        :param annotation:  labelme标注的数据
        :param shape: 图片shape(H,W,C),可进行坐标点的维度检查，避免越界
        :param task:  任务类型:det,obb,seg,pose
        :return:
        """
        h, w = shape[:2]
        bboxes, labels, points = [], [], []
        for anno in annotation:
            label = anno[0]
            datas = np.asarray(anno[1:])
            if task == "pose":
                cx, cy, cw, ch = datas[0:4] * (w, h, w, h)
                boxes = [cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2]
                polys = datas[4:].reshape(-1, 3)
                # polys = polys[polys[:, 2] > 0]  # 只选择可见的关键点
                polys = polys[:, 0:3] * (w, h, 1)
            elif len(anno) == 5 or task == "det":
                cx, cy, cw, ch = datas[0:4] * (w, h, w, h)
                boxes = [cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2]
                polys = image_utils.boxes2polygons([boxes])[0]
            else:
                polys = datas.reshape(-1, 2) * (w, h)
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


def save_yolo(out_root, image_file, labels, boxes=[], points=[], use_seg=True, image=None,
              vis=False, delay=0, thickness=2):
    """
    保存YOLO数据格式
    :param out_root: 输出根目录
    :param image_file: 图片路径
    :param labels:  目标label index
    :param boxes: 目标矩形框(xmin, ymin, xmax, ymax)
    :param points: 目标轮廓
    :param use_seg: 数据格式，True是YOLO实例分割数据格式 [class_index, cx, cy, w,  h]
                            False是YOLO目标检测格式 [class_index, x1, y1, x2, y2, x3, y3, x4, y4,....]

    :param image: 图像(np.ndarray)
    :param vis: 可视化标注效果
    :return:
    """
    if image is None: image = cv2.imread(image_file)
    if image is None:
        print("Error: image is None,image_file={},names={}".format(image_file, labels))
        return
    if len(points) == 0: return
    if vis:
        dst = image_utils.draw_image_contours(image.copy(), points, texts=labels, alpha=0.3, thickness=thickness)
        dst = image_utils.draw_image_boxes_texts(dst, boxes, texts=labels, thickness=thickness)
        dst = image_utils.show_image("image", dst, delay=delay)
    h, w = image.shape[:2]
    if use_seg:
        conts = [np.asarray(p) / (w, h) for p in points]
    else:
        conts = image_utils.xyxy2cxcywh(boxes).reshape(-1, 2, 2) / (w, h)
    image_name = os.path.basename(image_file)
    image_id, postfix = file_utils.split_postfix(image_name)
    anno_file = file_utils.create_dir(out_root, "labels", f"{image_id}.txt")
    file_path = file_utils.create_dir(out_root, "images", f"{image_name}")
    text_data = [[l] + [f"{s:3.5f}" for s in np.asarray(p).reshape(-1).tolist()] for l, p in zip(labels, conts)]
    file_utils.write_data(anno_file, text_data, split=" ")
    file_utils.copy_file(image_file, file_path)


def show_target_image(image, boxes, labels, points=[], class_name=None, use_rgb=True, thickness=2):
    """
    :param image:
    :param boxes:
    :param labels:
    :param points: [shape(n,2),[shape(n,2)]]
    :param class_name:
    :param use_rgb:
    :return:
    """
    dst = image.copy()
    if class_name: labels = [class_name[i] for i in labels]
    dst = image_utils.draw_image_boxes_texts(dst, boxes, texts=labels, thickness=thickness)
    dst = image_utils.draw_image_contours(dst, contours=points, texts=labels, alpha=0.3, thickness=thickness)
    dst = image_utils.image_hstack([image, dst])
    image_utils.cv_show_image("image", dst, use_rgb=use_rgb)


if __name__ == "__main__":
    class_name = ['AngelFish', 'BlueTang', 'ButterflyFish', 'ClownFish', 'GoldFish', 'Gourami', 'MorishIdol',
                  'PlatyFish', 'RibbonedSweetlips', 'ThreeStripedDamselfish', 'YellowCichlid', 'YellowTang',
                  'ZebraFish']
    # filename = "/home/dm/nasdata/dataset/csdn/helmet/helmet-dataset-v2/train.txt"
    # filename = "/home/dm/nasdata/dataset/csdn/helmet/helmet-asian/total.txt"
    # filename = "/home/dm/nasdata/dataset/csdn/helmet/helmet-asian/total.txt"
    data_root = "/home/PKing/nasdata/tmp/tmp/Fish/train/yolo"
    dataset = YOLODataset(filename=None,
                          data_root=data_root,
                          anno_dir=None,
                          image_dir=None,
                          class_name=class_name,
                          check=False,
                          phase="val",
                          shuffle=False)
    print("have num:{}".format(len(dataset)))
    for i in range(len(dataset)):
        # i = 16
        print(i)  # i=20
        data = dataset.__getitem__(i)
        image, boxes, names = data["image"], data["boxes"], data["names"]
        points = data["points"]
        h, w = image.shape[:2]
        image_file = data["image_file"]
        print(image_file)
        show_target_image(image, boxes, names, points=points)
