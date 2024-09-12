# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : panjq
# @Date   : 2019-9-20 13:18:34
# --------------------------------------------------------
"""

import os
import sys

sys.path.append(os.getcwd())
import PIL.Image as Image
import numpy as np
import random
import math
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset
from pybaseutils import image_utils, file_utils
from pybaseutils.dataloader import data_resample


class TextDataset(Dataset):

    def __init__(self, filename, data_root=None, class_name=None, transform=None, shuffle=False, use_rgb=False,
                 phase="test", disp=False, check=False, **kwargs):
        """
        :param filename:
        :param data_root:
        :param class_name:
        :param transform:
        :param shuffle:
        :param use_rgb:
        :param phase:
        :param disp:
        :param check:
        :param kwargs: use_max,use_mean,crop_scale,resample,save_info
        """
        self.kwargs = kwargs
        self.use_rgb = use_rgb
        self.transform = transform
        self.phase = phase
        self.class_name, self.class_dict = self.parser_classes(class_name)
        self.item_list = self.parser_dataset(filename, data_root=data_root, shuffle=shuffle, check=check)
        self.resample = kwargs.get("resample", False)
        if self.resample:
            self.data_resample = data_resample.DataResample(self.item_list,
                                                            label_index=1,
                                                            shuffle=shuffle,
                                                            disp=disp)
            self.item_list = self.data_resample.update(True)
            class_count = self.data_resample.class_count  # resample前，每个类别的分布
            balance_nums = self.data_resample.balance_nums  # resample后，每个类别的分布
        self.class_count = self.count_class_info(self.item_list, class_name=self.class_name)
        self.num_images = len(self.item_list)
        self.classes = list(self.class_dict.values())
        self.num_classes = max(self.classes) + 1
        self.info(save_info=kwargs.get("save_info", ""))

    def info(self, save_info=""):
        print("----------------------- {} DATASET INFO -----------------------".format(self.phase.upper()))
        print("Dataset have images   :{}".format(len(self.item_list)))
        print("Dataset class_name    :{}".format(self.class_name))
        print("Dataset class_dict    :{}".format(self.class_dict))
        print("Dataset class_count   :{}".format(self.class_count))
        print("Dataset num images    :{}".format(len(self.item_list)))
        print("Dataset num_classes   :{}".format(self.num_classes))
        print("Dataset resample      :{}".format(self.resample))
        if save_info:
            if not os.path.exists(save_info): os.makedirs(save_info)
            m = np.mean(list(self.class_count.values()))
            class_lack = {n: c for n, c in self.class_count.items() if c < m * 0.5}
            class_lack = sorted(class_lack.items(), key=lambda x: x[1], reverse=True)
            class_lack = {n[0]: n[1] for n in class_lack}
            class_lack.update({"mean": m})
            file_utils.write_json_path(os.path.join(save_info, f"{self.phase}_class_dict.json"), self.class_dict)
            file_utils.write_json_path(os.path.join(save_info, f"{self.phase}_class_count.json"), self.class_count)
            file_utils.write_list_data(os.path.join(save_info, f"{self.phase}_class_name.txt"), self.class_name)
            file_utils.write_json_path(os.path.join(save_info, f"{self.phase}_class_lack.json"), class_lack)
            print("loss_labels: {}".format(class_lack))
        print("------------------------------------------------------------------")

    def parser_dataset(self, filename, data_root="", shuffle=False, check=False):
        """
        保存格式：[path,label] 或者 [path,label,xmin,ymin,xmax,ymax]
        :param filename:
        :param data_root:
        :param shuffle:
        :param check:
        :return:
        """
        data_list = self.load_dataset(filename, data_root=data_root)
        if not self.class_name:
            self.class_name = list(set([d[1] for d in data_list]))
            self.class_name, self.class_dict = self.parser_classes(self.class_name)
        item_list = []
        for data in data_list:
            label = data[1]
            if label not in self.class_dict: continue
            data[1] = self.class_dict[label]
            item_list.append(data)
        if check: item_list = self.check_item(item_list)
        assert self.class_name, f"类别为空，请检查，class_name={self.class_name}"
        assert item_list, f"文件列表为空，请检查输入数据，filename={filename}"
        if shuffle:
            random.seed(100)
            random.shuffle(item_list)
        return item_list

    def load_dataset(self, filename, data_root=""):
        """
        保存格式：[path,label] 或者 [path,label,xmin,ymin,xmax,ymax]
        :param filename:
        :param data_root:
        :return: [path,label] 或者 [path,label,xmin,ymin,xmax,ymax]
        """
        if isinstance(filename, str): filename = [filename]
        item_list = []
        for file in filename:
            root = data_root if data_root else os.path.dirname(file)
            data = file_utils.read_data(file, split=",")
            data = [[os.path.join(root, d[0])] + d[1:] for d in data]
            item_list += data
        return item_list

    def parser_classes(self, class_name):
        """
        class_dict = {class_name: i for i, class_name in enumerate(class_name)}
        :param
        :return:
        """
        if isinstance(class_name, str):
            class_name = file_utils.read_data(class_name, split=None)
        if isinstance(class_name, list):
            class_dict = {name: i for i, name in enumerate(class_name)}
        elif isinstance(class_name, dict):
            class_dict = class_name
        else:
            class_dict = None
        return class_name, class_dict

    def check_item(self, item_list):
        """
        :param item_list:
        :return:
        """
        dst_list = []
        print("Please wait, it's in checking")
        for item in tqdm(item_list):
            file, label = item[0], item[1]
            if not os.path.exists(file):
                print("no file:{}".format(file))
                continue
            dst_list.append(item)
        print("have nums image:{},legal image:{}".format(len(item_list), len(dst_list)))
        return dst_list

    def __getitem__(self, index):
        """
        :param index:
        :return: {"image": image, "label": label}
        """
        item = self.item_list[index]
        bbox = item[2:] if len(item) == 6 else []
        image_file, label = item[0], item[1]
        image = self.read_image(image_file, use_rgb=self.use_rgb)
        image = self.crop_image(image, bbox=bbox, **self.kwargs) if bbox else image
        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)
        if image is None:
            index = int(random.uniform(0, self.num_images))
            return self.__getitem__(index)
        return {"image": image, "label": label}

    def __len__(self):
        if self.resample:
            self.item_list = self.data_resample.update(True)
        return len(self.item_list)

    def crop_image(self, image, bbox, **kwargs):
        """裁剪图片"""
        if not bbox: return image
        boxes = image_utils.get_square_boxes(boxes=[bbox],
                                             use_max=kwargs.get("use_max", True),
                                             use_mean=kwargs.get("use_mean", True))
        boxes = image_utils.extend_xyxy(boxes, scale=kwargs.get("crop_scale", []))
        image = image_utils.get_boxes_crop(image, boxes)[0]
        return image

    @staticmethod
    def read_image(path, use_rgb=True):
        """
        读取图片
        :param path:
        :param use_rgb:
        :return:
        """
        image = cv2.imread(path)
        if use_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
        return image

    @staticmethod
    def read_image_fast(path, use_rgb=True, use_fast=True, kb_th=100):
        """
        读取图片
        :param path:
        :param use_rgb:
        :param use_fast:
        :return:
        """
        size = file_utils.get_file_size(path) if use_fast else kb_th + 1
        if size > 2 * kb_th:
            image = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_4 | cv2.IMREAD_COLOR)
        elif size > kb_th:
            image = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_2 | cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(path, flags=cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            print("bad image:{}".format(path))
            return None
        if use_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
        return image

    @staticmethod
    def count_class_info(item_list, class_name=None, index=1):
        """
        统计类别信息
        item_list=[[filename,label,...],[filename,label,...]]
        :param item_list:
        :param class_name:
        :return:
        """
        count = {}
        for item in item_list:
            label = item[index]
            count[label] = count[label] + 1 if label in count else 1
        if class_name: count = {class_name[k]: v for k, v in count.items()}
        return count


if __name__ == '__main__':
    from pybaseutils import image_utils
    from torchvision import transforms

    filenames = [
        '/home/PKing/nasdata/tmp/tmp/RealFakeFace/anti-spoofing-images-v1/train.txt',
    ]
    class_name = None
    input_size = [112, 112]
    rgb_mean = [0., 0., 0.]
    rgb_std = [1.0, 1.0, 1.0]
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=rgb_mean, std=rgb_std),
    ])
    dataset = TextDataset(filename=filenames,
                          transform=transform,
                          class_name=class_name,
                          resample=False,
                          shuffle=True,
                          crop_scale=(1.5, 1.5),
                          disp=True)
    for i in range(len(dataset)):
        data_info = dataset.__getitem__(i)
        image, label = data_info["image"], data_info["label"]
        image = np.asarray(image).transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
        image = np.asarray(image * 255, dtype=np.uint8)
        label = np.asarray(label, dtype=np.int32)
        print("batch_image.shape:{},batch_label:{}".format(image.shape, label))
        image_utils.cv_show_image("image", image)
