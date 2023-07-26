# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-06-29 18:31:12
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
from pybaseutils.dataloader.dataset import Dataset


class LabelMeDataset(Dataset):

    def __init__(self,
                 filename=None,
                 data_root=None,
                 anno_dir=None,
                 image_dir=None,
                 class_name=None,
                 use_rgb=False,
                 shuffle=False,
                 check=False,
                 **kwargs):
        """
        :param filename:
        :param data_root:
        :param anno_dir:
        :param image_dir:
        :param transform:
        :param use_rgb:
        :param shuffle:
        """
        super(LabelMeDataset, self).__init__()
        self.min_area = 1 / 1000  # 如果前景面积不足0.1%,则去除
        self.use_rgb = use_rgb
        self.class_name, self.class_dict = self.parser_classes(class_name)
        parser = self.parser_paths(filename, data_root, anno_dir, image_dir)
        self.data_root, self.anno_dir, self.image_dir, self.image_id = parser
        self.postfix = self.get_image_postfix(self.image_dir, self.image_id)
        self.classes = list(self.class_dict.values()) if self.class_dict else None
        self.num_classes = max(list(self.class_dict.values())) + 1 if self.class_dict else None
        self.class_weights = None
        if check:
            self.image_id = self.checking(self.image_id)
        if shuffle:
            random.seed(200)
            random.shuffle(self.image_id)
        self.num_images = len(self.image_id)
        self.scale_rate = 1.0
        self.target_type = 'gaussian'
        self.sigma = 2
        print("Dataset class_name    :{}".format(class_name))
        print("Dataset class_dict    :{}".format(self.class_dict))
        print("Dataset num images    :{}".format(len(self.image_id)))
        print("Dataset num_classes   :{}".format(self.num_classes))

    def __len__(self):
        return len(self.image_id)

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
            class_name = super().read_files(class_name)
        elif isinstance(class_name, numbers.Number):
            class_name = [str(i) for i in range(int(class_name))]
        elif isinstance(class_name, list) and "unique" in class_name:
            self.unique = True
        if isinstance(class_name, list):
            class_dict = {str(class_name): i for i, class_name in enumerate(class_name)}
        elif isinstance(class_name, dict):
            class_dict = class_name
            class_name = list(class_dict.keys())
        else:
            class_dict = None
        return class_name, class_dict

    def get_image_postfix(self, image_dir, image_id):
        """
        获得图像文件后缀名
        :param image_dir:
        :return:
        """
        if "." in image_id[0]:
            postfix = ""
        else:
            image_list = glob.glob(os.path.join(image_dir, "*"))
            image_list = [file for file in image_list if not file.endswith("json")]
            postfix = os.path.basename(image_list[0]).split(".")[1]
        return postfix

    def get_image_anno_file(self, index):
        """
        :param index:
        :return:
        """
        image_id = self.index2id(index)
        image_file, annotation_file, image_id = self.__get_image_anno_file(self.image_dir,
                                                                           self.anno_dir,
                                                                           image_id,
                                                                           self.postfix)
        return image_file, annotation_file, image_id

    def __get_image_anno_file(self, image_dir, anno_dir, image_id: str, img_postfix):
        """
        :param image_dir:
        :param anno_dir:
        :param image_id:
        :param img_postfix:
        :return:
        """
        if not img_postfix and "." in image_id:
            img_postfix = image_id.split(".")[-1]
            image_id = image_id[:-len(img_postfix) - 1]
        image_file = os.path.join(image_dir, "{}.{}".format(image_id, img_postfix))
        annotation_file = os.path.join(anno_dir, "{}.json".format(image_id))
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
            box, label, point = self.parser_annotation(annotation, self.class_dict, None)
            if len(label) == 0:
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
        image_id = []
        if isinstance(filename, str):
            image_id = self.read_files(filename, split=",")
            data_root = os.path.dirname(filename)
        if not anno_dir:  # 如果anno_dir为空，则自动搜寻可能存在图片目录
            image_sub = ["json"]
            anno_dir = self.search_path(data_root, image_sub)
        if not image_dir:
            image_dir = self.search_path(data_root, ["JPEGImages", "images"])
        if anno_dir and not image_id:
            image_id = self.get_file_list(anno_dir, postfix=["*.json"], basename=True)
        elif image_dir and not image_id:
            image_id = self.get_file_list(anno_dir, postfix=image_utils.IMG_POSTFIX, basename=True)
        # assert os.path.exists(image_dir), Exception("no directory:{}".format(image_dir))
        # assert os.path.exists(anno_dir), Exception("no directory:{}".format(anno_dir))
        return data_root, anno_dir, image_dir, image_id

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
        box, label, point = self.parser_annotation(annotation, self.class_dict, shape)
        data = {"image": image, "point": point, "box": box, "label": label,
                "image_file": image_file, "anno_file": anno_file}
        return data

    @staticmethod
    def parser_annotation(annotation: dict, class_dict={}, shape=None):
        """
        :param annotation:  labelme标注的数据
        :param class_dict:  label映射
        :param shape: 图片shape(H,W,C),可进行坐标点的维度检查，避免越界
        :return:
        """
        bboxes, labels, points = [], [], []
        for anno in annotation:
            label = anno["label"].lower()
            if class_dict:
                if not label in class_dict:
                    continue
                if isinstance(class_dict, dict):
                    label = class_dict[label]
            pts = np.asarray(anno["points"], dtype=np.int32)
            if shape:
                h, w = shape[:2]
                pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            box = image_utils.polygons2boxes([pts])[0]
            labels.append(label)
            bboxes.append(box)
            points.append(pts)
        return bboxes, labels, points

    def index2id(self, index):
        """
        :param index: int or str
        :return:
        """
        if isinstance(index, numbers.Number):
            image_id = self.image_id[index]
        else:
            image_id = index
        return image_id

    def __len__(self):
        return len(self.image_id)

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
        try:
            image = cv2.imread(image_file, cv2.IMREAD_COLOR)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = image[:, :, 0:3]
            if use_rgb:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise Exception("empty image:{}".format(image_file))
        return image

    @staticmethod
    def load_annotations(ann_file: str):
        with open(ann_file, "r") as f: annotation = json.load(f)
        annos = annotation["shapes"]
        return annos


def parser_labelme(anno_file, class_dict={}, shape=None):
    """
    :param annotation:  labelme标注的数据
    :param class_dict:  label映射
    :param shape: 图片shape(H,W,C),可进行坐标点的维度检查，避免越界
    :return:
    """
    annotation = LabelMeDataset.load_annotations(anno_file)
    bboxes, labels, points = LabelMeDataset.parser_annotation(annotation, class_dict, shape)
    return bboxes, labels, points


def show_target_image(image, bboxes, labels, points):
    image = image_utils.draw_image_bboxes_text(image, bboxes, labels, color=(255, 0, 0))
    image = image_utils.draw_landmark(image, points, color=(0, 255, 0))
    image_utils.cv_show_image("det", image)


if __name__ == "__main__":
    from pybaseutils.maker import maker_labelme

    filename = "/home/dm/nasdata/dataset-dmai/handwriting/grid-det/grid_cross_points_v3/trainval.txt"
    # filename = None
    # anno_dir = "/media/dm/新加卷/SDK/base-utils/data/labelme"
    # image_dir = "/media/dm/新加卷/SDK/base-utils/data/labelme"
    input_size = [160, 160]
    dataset = LabelMeDataset(filename=filename,
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
        image, points, bboxes, labels = data["image"], data["point"], data["box"], data["label"]
        h, w = image.shape[:2]
        image_file = data["image_file"]
        anno_file = os.path.join("masker", "{}.json".format(os.path.basename(image_file).split(".")[0]))
        maker_labelme(anno_file, points, labels, (w, h), image_name=image_file, image_bs64=None)
        show_target_image(image, bboxes, labels, points)
