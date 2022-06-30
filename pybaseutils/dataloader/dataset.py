# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-06-29 18:58:33
    @Brief  :
"""
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-02-05 11:01:49
# --------------------------------------------------------
"""
import os
import numpy as np
import cv2
import glob
import random


class Dataset(object):
    """
    from torch.utils.data import Dataset,DataLoader, ConcatDataset
    """

    def __init__(self, **kwargs):
        self.image_id = []

    def __getitem__(self, index):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])

    def __len__(self):
        raise NotImplementedError

    @staticmethod
    def read_files(filename, split=None):
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
        self.image_id = []
        self.dataset = datasets
        self.shuffle = shuffle
        for dataset_id, dataset in enumerate(self.dataset):
            image_id = dataset.image_id
            image_id = self.add_dataset_id(image_id, dataset_id)
            self.image_id += image_id
            self.classes = dataset.classes
        if shuffle:
            random.seed(200)
            random.shuffle(self.image_id)

    def add_dataset_id(self, image_id, dataset_id):
        """
        :param image_id:
        :param dataset_id:
        :return:
        """
        out_image_id = []
        for id in image_id:
            out_image_id.append({"dataset_id": dataset_id, "image_id": id})
        return out_image_id

    def __getitem__(self, index):
        """
        :param index: int
        :return:
        """
        dataset_id = self.image_id[index]["dataset_id"]
        image_id = self.image_id[index]["image_id"]
        dataset = self.dataset[dataset_id]
        data = dataset.__getitem__(image_id)
        return data

    def get_image_anno_file(self, index):
        dataset_id = self.image_id[index]["dataset_id"]
        image_id = self.image_id[index]["image_id"]
        return self.dataset[dataset_id].get_image_anno_file(image_id)

    def get_annotation(self, xml_file):
        return self.dataset[0].get_annotation(xml_file)

    def read_image(self, image_file):
        return self.dataset[0].read_image(image_file, use_rgb=self.dataset[0].use_rgb)

    def __len__(self):
        return len(self.image_id)
