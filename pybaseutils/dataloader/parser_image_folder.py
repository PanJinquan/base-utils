# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : panjq
# @Date   : 2019-9-20 13:18:34
# --------------------------------------------------------
"""

import os
import cv2
import PIL.Image as Image
import numpy as np
import random
from pybaseutils import image_utils, file_utils
from pybaseutils.dataloader import parser_image_text


class FolderDataset(parser_image_text.TextDataset):
    def __init__(self, image_dir, class_name=None, transform=None, use_rgb=False, shuffle=False,
                 phase="test", disp=False, check=False, **kwargs):
        """
        :param image_dir: [image_dir]->list or `path/to/image_dir`->str
        :param class_name:
        :param transform: torch transform
        :param shuffle:
        :param disp:
        """
        super(FolderDataset, self).__init__(filename=image_dir,
                                            data_root=None,
                                            class_name=class_name,
                                            transform=transform,
                                            shuffle=shuffle,
                                            use_rgb=use_rgb,
                                            phase=phase,
                                            disp=disp,
                                            check=check,
                                            **kwargs)

    def __getitem__(self, index):
        """
        :param index:
        :return: {"image": image, "label": label}
        """
        item = self.item_list[index]
        bbox = item[2:] if len(item) == 6 else []
        image_file, label = item[0], item[1]
        image = self.read_image(image_file, use_rgb=self.use_rgb)
        image = self.crop_image(image, bbox=bbox) if bbox else image
        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)
        if image is None:
            index = int(random.uniform(0, self.num_images))
            return self.__getitem__(index)
        return {"image": image, "label": label}

    def load_dataset(self, filename, data_root="", use_sub=False):
        """
        保存格式：[path,label] 或者 [path,label,xmin,ymin,xmax,,ymax]
        :param filename:
        :param data_root:
        :return: item_list [path,label] 或者 [path,label,xmin,ymin,xmax,,ymax]
        """
        if isinstance(filename, str): filename = [filename]
        item_list = []
        for i, dir in enumerate(filename):
            print("loading image from:{}".format(dir))
            if not os.path.exists(dir): raise Exception("文件不存在，image_dir:{}".format(dir))
            paths, labels = file_utils.get_files_labels(dir, postfix=file_utils.IMG_POSTFIX)
            # TODO # 避免多个数据集的相同的label
            if use_sub:  labels = [os.path.join(str(i), l) for l in labels]
            data = [[p, l] for p, l in zip(paths, labels)]
            item_list += data
        return item_list


if __name__ == '__main__':
    from pybaseutils import image_utils
    from torchvision import transforms

    image_dir = ['/home/PKing/nasdata/release/infrastructure/DMClassification/data/dataset/train']
    class_name = None
    input_size = [224, 224]
    rgb_mean = [0., 0., 0.]
    rgb_std = [1.0, 1.0, 1.0]
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=rgb_mean, std=rgb_std),
    ])
    dataset = FolderDataset(image_dir=image_dir,
                            transform=transform,
                            shuffle=True,
                            class_name=class_name,
                            resample=True,
                            disp=True)
    for i in range(len(dataset)):
        data_info = dataset.__getitem__(i)
        image, label = data_info["image"], data_info["label"]
        image = np.asarray(image).transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
        image = np.asarray(image * 255, dtype=np.uint8)
        label = np.asarray(label, dtype=np.int32)
        print("batch_image.shape:{},batch_label:{}".format(image.shape, label))
        image_utils.cv_show_image("image", image)
