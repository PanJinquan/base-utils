# -*- coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail :
    @Date   : 2023-08-10 10:18:32
    @Brief  :
"""
import os
from fileinput import filename

import numpy as np
import cv2
import glob
import random
import numbers
import json
import pandas as pd
from streamlit import video
from tqdm import tqdm
from pybaseutils import image_utils, file_utils, json_utils, text_utils
from pybaseutils.dataloader.base_dataset import Dataset, ConcatDataset
from pybaseutils import json_utils, pandas_utils, image_utils
from pybaseutils.cvutils import video_utils


class VIADataset(Dataset):

    def __init__(self,
                 video_dir=None,
                 anno_dir=None,
                 class_name=None,
                 use_rgb=False,
                 shuffle=False,
                 check=False,
                 min_points=-1,
                 **kwargs):
        """
        dataset.image_ids
        dataset.classes
        dataset.class_name
        要求该目录下存在video和csv
        data_root，anno_dir只要存在一个即可，程序会自动搜索video和csv
        :param video_dir:
        :param anno_dir:
        :param class_name: 当class_name=None且check=True,将自动获取所有class,当class_name=[]会直接返回name
        :param use_rgb:
        :param shuffle:
        :param check: 当class_name=None且check=True,将自动获取所有class
        :param min_points: 当标注的轮廓点的个数小于min_points，会被剔除；负数不剔除
        :param kwargs: read_image: 是否读取图片，否则image=None
        """
        super(VIADataset, self).__init__()
        self.min_area = 1 / 1000  # 如果前景面积不足0.1%,则去除
        self.use_rgb = use_rgb
        self.min_points = min_points
        self.kwargs = kwargs
        self.class_name, self.class_dict = self.parser_classes(class_name)
        self.video_dir, self.anno_dir, self.image_ids = self.parser_paths(video_dir, anno_dir)
        self.classes = list(self.class_dict.values()) if self.class_dict else None
        self.class_weights = None
        # self.num_classes = max(list(self.class_dict.values())) + 1 if self.class_dict else None
        if check:
            self.image_ids = self.checking(self.image_ids)
        if shuffle:
            random.seed(200)
            random.shuffle(self.image_ids)
        self.num_images = len(self.image_ids)
        print("VIADataset anno_dir      :{}".format(self.anno_dir))
        print("VIADataset video_dir     :{}".format(self.video_dir))
        print("VIADataset class_name    :{}".format(self.class_name))
        print("VIADataset class_dict    :{}".format(self.class_dict))
        print("VIADataset num videos    :{}".format(len(self.image_ids)))
        # print("VIADataset num_classes   :{}".format(self.num_classes))
        print("------" * 10)

    def __len__(self):
        return len(self.image_ids)

    def parser_paths(self, video_dir, anno_dir=None):
        """
        :param video_dir:
        :param anno_dir:
        :return:
        """
        image_ids = self.get_file_list(video_dir, postfix=file_utils.VIDEO_POSTFIX, basename=False)
        image_ids = [os.path.basename(f) for f in image_ids]
        if not anno_dir: anno_dir = video_dir
        assert len(image_ids) > 0, f"image_ids is empty,video_dir={video_dir},anno_dir={anno_dir}"
        return video_dir, anno_dir, image_ids

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

    def get_image_anno_file(self, index):
        """
        :param index:
        :return:
        """
        image_id = self.index2id(index)
        video_file, anno_file, image_id = self.__get_image_anno_file(self.video_dir, self.anno_dir, image_id)
        return video_file, anno_file, image_id

    def __get_image_anno_file(self, video_dir, anno_dir, image_name: str):
        """
        :param video_dir:
        :param anno_dir:
        :param image_name:
        :param img_postfix:
        :return:
        """
        image_file = os.path.join(video_dir, image_name)
        img_postfix = image_name.split(".")[-1]
        image_id = image_name[:-len(img_postfix) - 1]
        anno_file = os.path.join(anno_dir, "{}.csv".format(image_id))
        return image_file, anno_file, image_name

    def checking(self, image_ids: list, ignore_empty=True):
        """
        :param image_ids:
        :param ignore_empty : 是否去除一些空数据
        :return:
        """
        print("Please wait, it's in checking")
        dst_ids = []
        for image_id in tqdm(image_ids):
            video_file, anno_file, image_id = self.get_image_anno_file(image_id)
            if not os.path.exists(anno_file):
                continue
            if not os.path.exists(video_file):
                continue
            dst_ids.append(image_id)
        return dst_ids

    def __getitem__(self, index):
        """
        :param index: int or str
        :return:rgb_image
        """
        image_id = self.index2id(index)
        video_file, anno_file, image_id = self.get_image_anno_file(image_id)
        labels, slices = self.load_annotations(anno_file)
        # TODO dict(boxes, labels, points, groups, names, keypoints)
        data_info = {"labels": labels, "slices": slices, "video_file": video_file, "anno_file": anno_file}
        return data_info

    def load_annotations(self, anno_file, sep=","):
        """
        :param anno_file:
        :param sep:分隔符
        :return:
        """
        names = ["name", "file_list", "temporal_segment_start", "temporal_segment_end", "metadata"]
        file = pd.read_csv(anno_file, sep=sep, names=names, comment="#")
        df = pd.DataFrame(file)
        labels = df['metadata'].tolist()
        slices = df[['temporal_segment_start', 'temporal_segment_end']].values.tolist()  # 时间片段
        labels = [json_utils.str2dict(s)["TEMPORAL-SEGMENTS"] for s in labels]
        assert len(labels) == len(slices), f"数据标注有问题：{anno_file}"
        return labels, slices

    def get_video_label(self, t, labels, slices, default="face", offset=(0, 0)):
        """
        获得视频在时刻t时的label
        :param t: 视频时刻
        :param labels:时间片段label集合
        :param slices: 时间片段集合
        :param default:  未标注的视频片段的默认label
        :param offset: 对标注的视频片段进行偏移offset=(左边界,右边界)
        :return: label视频片段的标注label, dist距离最近label的时间差(秒)
        """
        c = -1
        dist = np.inf
        for i, seg in enumerate(slices):
            t0 = seg[0] + offset[0]
            t1 = seg[1] - offset[1]
            if t0 < t < t1:
                dist = min(abs(t - t0), abs(t1 - t))
                c = i
                break
        label = labels[c] if c >= 0 else default
        return label, dist

    def video_extracter_example(self, output=None, vis=True, delay=30):
        """
        视频抽帧样例
        :param output:
        :param vis:
        :param delay:
        :return:
        """
        for i in range(len(self)):
            data_info = self.__getitem__(i)
            video_file, annot_file = data_info['video_file'], data_info['anno_file']
            self.video_extracter(video_file, annot_file, output=output, vis=vis, delay=delay)

    def video_extracter(self, video_file, annot_file, output=None, vis=True, delay=30):
        """
        根据标注文件进行视频抽帧
        :param video_file:
        :param annot_file:
        :param output:
        :param vis:
        :param delay:
        :return:
        """
        print(video_file)
        print(annot_file)
        video = video_utils.video_iterator(video_file, save_video=None, interval=15, vis=False, delay=delay)
        name = os.path.basename(video_file).split(".")[0]
        labels, slices = self.load_annotations(annot_file)
        index = 0
        for data_info in video:
            index = index + 1
            count = data_info["count"]
            frame = data_info["frame"]
            vtime = count / data_info['fps']  # 视频时刻
            if count <= 0: continue
            label, dist = self.get_video_label(vtime, labels, slices)
            dist = dist * data_info['fps']
            if label == "face" and index % 3 > 0: continue
            if label == "face" and dist < 30: continue  # TODO 为避免label定义模糊，跳过边界附近的图片
            if label == "低头" and dist < 8: continue  # TODO 为避免label定义模糊，跳过边界附近的图片
            if output:
                outfile = os.path.join(output, label, f"{name}_{count:0=4d}.jpg")
                file_utils.create_file_path(outfile)
                cv2.imwrite(outfile, frame)
            if vis:
                frame = image_utils.draw_text(frame, point=(10, 50), text=label, drawType="chinese")
                data_info["frame"] = frame


if __name__ == '__main__':
    video_dir = "/home/PKing/nasdata/tmp/tmp/RealFakeFace/living/driving"
    output = video_dir + "-frame"
    via = VIADataset(video_dir)
    # via.video_extracter(video_file, annot_file, output=output)
    via.video_extracter_example(output=output)
