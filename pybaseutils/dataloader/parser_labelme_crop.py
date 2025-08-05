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
import PIL.Image as Image
import numpy as np
import random
import cv2
import torch
import copy
from tqdm import tqdm
from typing import Dict, List
from pybaseutils import image_utils, file_utils
from pybaseutils.dataloader import parser_labelme, parser_image_text


class LabelmeDataset(parser_image_text.TextDataset):
    """加载Labelme数据并进行裁剪,图片和json文件都保存在images目录下"""

    def __init__(self, filename, class_name=None, transform=None, use_rgb=False, shuffle=False,
                 phase="test", disp=False, check=False, **kwargs):
        """
        :param filename:
        :param class_name:
        :param transform:
        :param use_rgb:
        :param shuffle:
        :param phase:
        :param disp:
        :param check:
        :param kwargs:  use_max,use_mean,crop_scale
        """
        self.tag = self.__class__.__name__
        self.dataset = None
        super(LabelmeDataset, self).__init__(data_file=filename,
                                             data_root=None,
                                             class_name=class_name,
                                             transform=transform,
                                             shuffle=shuffle,
                                             use_rgb=use_rgb,
                                             phase=phase,
                                             disp=disp,
                                             check=check,
                                             label_index="label",
                                             **kwargs)
        self.log("{:15s} have images:{},have samples:{}".format(self.tag, len(self.item_list), self.num_samples))

    def parser_dataset(self, data_file, data_root="", label_index="label", shuffle=False, check=False):
        """
        获得Labelme所有目标信息
        :param dataset:
        :return: [{"file","label","name","bbox"}],bbox非必须
        """
        self.dataset = parser_labelme.LabelMeDatasets(filename=None,
                                                      data_root=None,
                                                      anno_dir=data_file,
                                                      image_dir=None,
                                                      class_name=self.class_name,
                                                      check=False,
                                                      phase=self.phase,
                                                      use_rgb=self.use_rgb,
                                                      shuffle=self.shuffle,
                                                      read_image=False)
        item_list = []
        for index in tqdm(range(len(self.dataset)), desc="process data"):
            info = self.dataset.__getitem__(index)
            file, boxes = info["image_file"], info.get("boxes", [])
            for i in range(len(boxes)):
                if len(boxes[i]) == 0: continue
                item_list.append(dict(file=file,
                                      bbox=boxes[i],
                                      label=info["labels"][i],
                                      name=info["names"][i],
                                      point=info["points"][i],
                                      )
                                 )
        assert len(item_list) > 0, f"item_list is empty, check your data_file is ={data_file}"
        if not self.class_name:
            self.class_name = list(set([d['name'] for d in item_list]))
            self.class_name, self.class_dict = self.parser_classes(self.class_name)
        for i in range(len(item_list)):
            info = item_list[i]
            info.update(label=self.class_dict[info['name']])
        return item_list

    def __getitem__(self, index):
        """
        :param index:
        :return: {"image": image, "label": label}
        """
        item: dict = copy.deepcopy(self.item_list[index])  # TODO Fix a bug: 避免修改原始数据
        file, bbox = item["file"], item.get("bbox", [])
        image = self.read_image(file, use_rgb=self.use_rgb)
        image = self.crop_image(image, bbox=bbox, crop_scale=self.crop_scale, **self.kwargs)
        if self.transform:
            image = self.transform(Image.fromarray(image))
        if image is None:
            index = int(random.uniform(0, len(self)))
            return self.__getitem__(index)
        return dict(file=file, image=image, label=item["label"], name=item["name"])


if __name__ == '__main__':
    from torchvision import transforms

    filename = [
        "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-person-action/train-v2/高杆区4/dataset-v20/images",
        "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-person-action/train-v2/高杆区4/dataset-v21/images",
        "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-person-action/train-v2/高杆区4/dataset-v22/images",
        "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-person-action/train-v2/高杆区4/dataset-v23/images",
        "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-person-action/train-v2/高杆区4/dataset-v24/images"
    ]
    batch_size = 1
    crop_scale = (1.2, 1.2)
    input_size = [224, 224]
    rgb_mean = [0., 0., 0.]
    rgb_std = [1.0, 1.0, 1.0]
    class_name = None
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=rgb_mean, std=rgb_std),
    ])

    dataset = LabelmeDataset(filename=filename,
                             transform=transform,
                             resample=True,
                             shuffle=False,
                             class_name=class_name,
                             use_rgb=False,
                             disp=True,
                             crop_scale=crop_scale,
                             vis=False)
    for i in range(len(dataset)):
        data_info = dataset.__getitem__(i)
        file = data_info["file"]
        image, label, name = data_info["image"], data_info["label"], data_info["name"]
        image = np.asarray(image).transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
        image = np.asarray(image * 255, dtype=np.uint8)
        label = np.asarray(label, dtype=np.int32)
        print("{},image.shape:{},label:{},name:{}".format(file, image.shape, label, name))
        image_utils.cv_show_image("image", image)
