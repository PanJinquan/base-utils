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
import sys

sys.path.append(os.getcwd())
import PIL.Image as Image
import numpy as np
import random
import cv2
from tqdm import tqdm
from pybaseutils.dataloader.base_dataset import Dataset
from pybaseutils import image_utils, file_utils, json_utils
from pybaseutils.dataloader import data_resample


class TextDataset(Dataset):

    def __init__(self, data_file, data_root=None, class_name=None, transform=None, shuffle=False, use_rgb=False,
                 phase="test", disp=False, check=False, **kwargs):
        """
        数据文件保存格式：[path,label] 或者 [path,label,xmin,ymin,xmax,ymax]
        :param data_file: 数据文件路径，List[str] or str
        :param data_root: 数据文件根目录
        :param class_name: 类别文件/列表/字典
        :param transform:
        :param shuffle:
        :param use_rgb:
        :param phase:
        :param disp:
        :param check:
        :param kwargs:  log: print or log.info
                        use_max,
                        use_mean,
                        crop_scale,
                        resample,
                        save_info,
                        interval
        """
        self.tag = self.__class__.__name__
        self.data_file = data_file
        self.data_root = data_root
        self.use_rgb = use_rgb
        self.transform = transform
        self.phase = phase
        self.train = self.phase.lower() == "train"
        self.shuffle = shuffle
        self.check = check
        self.kwargs = kwargs
        self.log = kwargs.get('log', print) if kwargs.get('log', print) else print
        self.label_index = kwargs.get("label_index", "label")  # 类别字段key
        self.crop_scale = kwargs.pop("crop_scale", [])  # TODO bbox缩放系数
        self.class_name, self.class_dict = self.parser_classes(class_name)
        self.item_list = self.parser_dataset(data_file, data_root=data_root, label_index=self.label_index,
                                             shuffle=shuffle, check=check)
        self.resample = kwargs.get("resample", False)
        self.interval = kwargs.get('interval', 30)  # TODO 重采样间隔，低于该时间的不进行重采集，避免频繁采样
        if self.resample:
            self.data_resample = data_resample.DataResample(self.item_list,
                                                            label_index=self.label_index,
                                                            shuffle=shuffle,
                                                            disp=disp,
                                                            class_name=self.class_name,
                                                            interval=self.interval,
                                                            log=self.log)
            self.item_list = self.data_resample.update(True)
            src_class_count = self.data_resample.src_class_count  # resample前，每个类别的分布
            dst_class_count = self.data_resample.dst_class_count  # resample后，每个类别的分布
        self.class_count = self.count_class_info(self.item_list, class_name=self.class_name,
                                                 label_index=self.label_index)
        self.classes = list(self.class_dict.values())
        self.num_classes = max(self.classes) + 1
        self.num_samples = len(self.item_list)
        if self.log: self.info(save_info=kwargs.get("save_info", ""))

    def __len__(self):
        if self.resample:
            self.item_list = self.data_resample.update(True)
        return len(self.item_list)

    def info(self, save_info=""):
        self.log("----------------------- {} DATASET INFO -----------------------".format(self.phase.upper()))
        self.log("{:15s} kwargs        :{}".format(self.tag, self.kwargs))
        self.log("{:15s} num_samples   :{}".format(self.tag, len(self.item_list)))
        self.log("{:15s} num_classes   :{}".format(self.tag, self.num_classes))
        self.log("{:15s} class_name    :{}".format(self.tag, self.class_name))
        self.log("{:15s} class_dict    :{}".format(self.tag, self.class_dict))
        self.log("{:15s} class_count   :{}".format(self.tag, self.class_count))
        self.log("{:15s} resample      :resample={},interval={}".format(self.tag, self.resample, self.interval))
        if save_info:
            if not os.path.exists(save_info): os.makedirs(save_info)
            m = np.mean(list(self.class_count.values()))
            class_lack = {n: c for n, c in self.class_count.items() if c < m * 0.5}
            class_lack = sorted(class_lack.items(), key=lambda x: x[1], reverse=True)
            class_lack = {n[0]: n[1] for n in class_lack}
            class_lack.update({"mean": m})
            file_utils.save_json(os.path.join(save_info, f"{self.phase}_class_dict.json"), self.class_dict)
            file_utils.save_json(os.path.join(save_info, f"{self.phase}_class_count.json"), self.class_count)
            file_utils.write_list_data(os.path.join(save_info, f"{self.phase}_class_name.txt"), self.class_name)
            file_utils.save_json(os.path.join(save_info, f"{self.phase}_class_lack.json"), class_lack)
            self.log("loss_labels: {}".format(class_lack))
        self.log("------------------------------------------------------------------")

    def parser_dataset(self, data_file, data_root="", label_index="label", shuffle=False, check=False):
        """
        保存格式：[path,label] 或者 [path,label,xmin,ymin,xmax,ymax]
        :param data_file: List([])
        :param data_root:
        :param label_index: label index
        :param shuffle:
        :param check:
        :return:
        """
        data_list = self.load_dataset(data_file, data_root=data_root)
        if not self.class_name:
            self.class_name = list(set([d[label_index] for d in data_list]))
            self.class_name = sorted(self.class_name)
            self.class_name, self.class_dict = self.parser_classes(self.class_name)
        item_list = []
        for data in data_list:
            label = data[label_index]
            if label not in self.class_dict: continue
            data[label_index] = self.class_dict[label]
            item_list.append(data)
        if check: item_list = self.check_item(item_list)
        assert self.class_name, f"类别为空，请检查，class_name={self.class_name}"
        assert item_list, f"文件列表为空，请检查输入数据，data_file={data_file}"
        if shuffle:
            random.seed(100)
            random.shuffle(item_list)
        return item_list

    def load_dataset(self, data_file, data_root="", **kwargs):
        """
        txt保存格式：[path,name] 或者 [path,name,xmin,ymin,xmax,ymax]
        :param data_file:
        :param data_root:
        :return: item_list [{"file","label","name","bbox"}],bbox非必须
        """
        if isinstance(data_file, str): data_file = [data_file]
        item_list = []
        for file in data_file:
            root = data_root if data_root else os.path.dirname(file)
            content = file_utils.read_data(file, split=",")
            data = []
            for line in content:
                item = {"file": os.path.join(root, line[0]), "label": line[1], 'name': line[1]}
                if len(line) == 6: item['bbox'] = line[2:]  # (xmin,ymin,xmax,ymax)
                data.append(item)
            self.log("{:15s} loading data from:{},have {}".format(self.tag, file, len(data)))
            item_list += data
        return item_list

    def check_item(self, item_list):
        """
        :param item_list:
        :return:
        """
        dst_list = []
        for item in tqdm(item_list, desc="check data"):
            file, label, bbox = item["file"], item[self.label_index], item.get("bbox", [])
            if not os.path.exists(file):
                print("no file:{}".format(file))
                continue
            dst_list.append(item)
        self.log("{:15s} have nums samples:{},legal samples:{}".format(self.tag, len(item_list), len(dst_list)))
        return dst_list

    def __getitem__(self, index):
        """
        :param index:
        :return: {"image","label","name"}
        """
        item = self.item_list[index]
        file, label, name, bbox = item["file"], item[self.label_index], item['name'], item.get("bbox", [])
        image = self.read_image(file, use_rgb=self.use_rgb)
        image = self.crop_image(image, bbox=bbox, crop_scale=self.crop_scale, **self.kwargs)
        if self.transform:
            image = self.transform(Image.fromarray(image))
        if image is None:
            index = int(random.uniform(0, self.num_samples))
            return self.__getitem__(index)
        return dict(image=image, label=label, file=file, name=name)

    def crop_image(self, image, bbox, crop_scale=[], use_max=False, use_mean=True, **kwargs):
        """
        裁剪图片
        :param image:
        :param bbox:
        :param kwargs: use_max,use_mean,crop_scale
                use_max: 是否按照每个box(w,h)最大值(True)/最小值(False)进行转换(默认)，可以保证目标裁剪完整
                use_mean: 是否按照每个box(w,h)平均值进行转换(优先级比use_max高),但当目标长宽比比较大时，可能出现目标裁剪不完整的问题
        :return:
        """
        if len(bbox) == 0: return image
        boxes = image_utils.get_square_boxes(boxes=[bbox], use_max=use_max, use_mean=use_mean)
        boxes = image_utils.extend_xyxy(boxes, scale=crop_scale)
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


if __name__ == '__main__':
    from pybaseutils import image_utils
    from torchvision import transforms

    data_files = [
        '/home/PKing/nasdata/tmp/tmp/RealFakeFace/anti-spoofing/anti-spoofing-images-v2/test.txt',
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
    class_name = ['fake', 'real']
    dataset = TextDataset(data_file=data_files,
                          transform=transform,
                          class_name=class_name,
                          resample=True,
                          shuffle=True,
                          check=False,
                          crop_scale=(1.5, 1.5),
                          disp=False)
    for i in range(len(dataset)):
        data_info = dataset.__getitem__(i)
        image, label, file = data_info["image"], data_info["label"], data_info["file"]
        image = np.asarray(image).transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
        image = np.asarray(image * 255, dtype=np.uint8)
        label = np.asarray(label, dtype=np.int32)
        print("{},batch_image.shape:{},batch_label:{}".format(file, image.shape, label))
        image_utils.cv_show_image("image", image)
