# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 390737991@qq.com
# @Date   : 2020-02-05 11:01:49
# --------------------------------------------------------
"""
import os
import numpy as np
import cv2
import glob
import random
import numbers
import torch
import torchvision.transforms.functional as F
import PIL.Image as Image
from tqdm import tqdm
from pybaseutils.dataloader.base_dataset import Dataset, ConcatDataset, get_targets
from pybaseutils.dataloader.parser_labelme import LabelMeDatasets
from pybaseutils.transforms import imgaug_utils
from pybaseutils import image_utils, file_utils


class LabelmeLineDataset():
    # TODO 直线解析器
    def __init__(self,
                 filename=None,
                 data_root=None,
                 anno_dir=None,
                 image_dir=None,
                 class_name=None,
                 transform=None,
                 phase="train",
                 use_rgb=True,
                 shuffle=False,
                 resample=False,
                 check=False,
                 **kwargs):
        """
        :param filename:
        :param data_root:
        :param anno_dir:
        :param image_dir:
        :param transform:
        :param use_rgb:
        :param keep_difficult:
        :param shuffle:
        """
        self.tag = self.__class__.__name__
        assert phase in ["train", "val", "test"]
        self.lines_name = ["布线通道"]  # 直线
        self.point_name = ["胶枪头与布线通道无接触", "胶枪头与布线通道有接触"]  # 胶枪头
        self.class_weight = [1.0, 2.0]  # (直线, 胶枪头) 权重
        self.label_maps = {0: "其他", 1: "胶枪头与布线通道无接触", 2: "胶枪头与布线通道有接触"}
        self.log = kwargs.get('log', print) if kwargs.get('log', print) else print
        self.phase = phase
        self.min_area = 1 / 1000  # 如果前景面积不足0.1%,则去除
        self.transform = transform
        self.dataset = LabelMeDatasets(filename=filename,
                                       data_root=data_root,
                                       anno_dir=anno_dir,
                                       image_dir=image_dir,
                                       class_name=class_name,
                                       phase=phase,
                                       use_rgb=use_rgb,
                                       shuffle=shuffle,
                                       check=check,
                                       **kwargs)
        self.log("{:15s} dataset file           :{}".format(self.tag, anno_dir))
        self.log("{:15s} dataset samples nums   :{}".format(self.tag, len(self.dataset)))
        self.log("{:15s} dataset lines_name     :{}".format(self.tag, self.lines_name))
        self.log("{:15s} dataset point_name     :{}".format(self.tag, self.point_name))
        self.log("{:15s} dataset label_maps     :{}".format(self.tag, self.label_maps))
        self.log("{:15s} dataset class_weight   :{}".format(self.tag, self.class_weight))
        self.log("------" * 10)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        :param index: int or str
        :return:image: 返回torch.float32
                alpha：返回torch.float32，且归一化0~1
                trimap：返回torch.float32，且归一化0~1
        """
        data_info = self.dataset.__getitem__(index)
        image = data_info["image"]
        lines_info = get_targets(data_info, targets=self.lines_name, key='names', keys=['points', 'names'])
        point_info = get_targets(data_info, targets=self.point_name, key='names', keys=['points', 'names'])
        if self.transform:
            contours = lines_info.get("points", []) + point_info.get("points", [])
            contours = imgaug_utils.decode_polygons(contours, image.shape)
            image, contours = self.transform(image=image, polygons=contours)
            contours = imgaug_utils.encode_polygons(contours)
            lines_info["points"] = contours[:len(lines_info["points"])]
            point_info["points"] = contours[len(lines_info["points"]):]
        alpha0, alpha1, label = self.get_targets_heatmap(image, lines_info, point_info)
        # alpha = np.max([alpha1, alpha2], axis=0)
        target = np.concatenate([alpha0[None, :], alpha1[None, :]])
        weight = self.get_weight(target, class_weight=self.class_weight)
        # mask = torch.from_numpy(np.array(mask)).long()
        target = torch.from_numpy(target)
        weight = torch.from_numpy(weight)
        image = torch.from_numpy(image.transpose(2, 0, 1))  # HWC->CHW
        data = {"image": image, "target": target, "weight": weight}
        return data

    def get_weight(self, masks, class_weight, thresh=1):
        """
        :param masks:
        :param class_weight:
        :param thresh:
        :return:
        """
        # TODO w=0,相当于不计算该样本的Loss时，有利于预测未知的区域，提高召回率
        #      w=v，则相当于对进行缩放：Loss*v，可有效减少误检测，提高检测精度
        weight = []
        for i, mask in enumerate(masks):
            # w = 1.0 if np.sum(mask) > thresh else 0.0 #
            w = 1.0
            weight.append(w * class_weight[i])
        weight = np.asarray(weight, dtype=np.float32)
        return weight

    def get_targets_heatmap(self, image, lines_info: dict, point_info: dict):
        size = image.shape[:2][::-1]
        lines_alpha = self.get_lines_heatmap(lines_info.get("points", []), size=size, sigma=4)
        point_alpha = self.get_point_heatmap(point_info.get("points", []), size=size, sigma=10)
        names = point_info.get("names", [])
        if names and self.point_name[0] in names:  # TODO 胶枪头与布线通道无接触
            label = 1
        elif names and self.point_name[1] in names:  # TODO 胶枪头与布线通道有接触
            label = 2
        else:
            label = 0
        return lines_alpha, point_alpha, label

    @staticmethod
    def get_lines_heatmap(points, size, sigma=5, thickness=1):
        """
        使用 OpenCV绘制曲线并生成热力图
        参数:
            points: (N,nums, 2) array or list of [x, y] points
            size: (width, height) 输出图像尺寸
            sigma: 高斯热力图标准差
            thickness: 绘制曲线的线宽（建议=1）
        返回:
            heatmap: (height, width) float32 热力图
        """
        width, height, = size
        # 1. 创建空白二值掩码（全黑）
        mask = np.zeros((height, width), dtype=np.uint8)
        # 2. 将曲线点转换为 int32 并绘制连线
        for pts in points:
            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=thickness)
        # 3. 计算每个像素到最近白色像素（曲线）的距离,使用 cv2.distanceTransform 计算 L2 距离（欧氏距离）
        dist = cv2.distanceTransform(255 - mask, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)
        # 4. 转换为高斯热力图: exp(-dist^2 / (2*sigma^2))
        heatmap = np.exp(- (dist ** 2) / (2 * sigma ** 2))
        heatmap = heatmap.astype(np.float32)
        return heatmap

    @staticmethod
    def get_point_heatmap(points, size, sigma=2):
        """
        生成关键点热力图（每个关键点一个通道）
        参数:
            points (list of tuples): 关键点列表，每个元素为 (x, y)，注意x对应width，y对应height
            sigma (float): 高斯分布的标准差，控制热力图扩散范围
        返回:
            heatmaps: numpy array, shape = (num_keypoints, height, width)
        """
        width, height = size
        heatmaps = np.zeros((height, width), dtype=np.float32)
        # 为每个关键点生成一个热力图
        for n, pts in enumerate(points):
            pts = [np.asarray(pts)[0, :]]  # TODO Lableme标注的圆,圆心是pts[0],圆边的点是pts[1]
            for i, (x, y) in enumerate(pts):
                if x < 0 or y < 0 or x >= width or y >= height:
                    # 如果关键点在图像外，跳过（或可置为全0）
                    continue
                # 创建坐标网格
                y_grid, x_grid = np.ogrid[:height, :width]
                # 计算每个像素到关键点的欧氏距离的平方
                dist_sq = (x_grid - x) ** 2 + (y_grid - y) ** 2
                dist_sq = dist_sq.astype(np.float32)
                # 生成2D高斯热力图
                heatmap = np.exp(-dist_sq / (2 * sigma ** 2))
                heatmaps = np.max([heatmap, heatmaps], axis=0)
        return heatmaps


def show_target_image(image, alpha, normal=False, transpose=False):
    import numpy as np
    from pybaseutils import image_utils, color_utils
    image = np.asarray(image)
    alpha = np.asarray(alpha)
    print("image:{},alpha :{}".format(image.shape, alpha.shape))
    shape = alpha.shape
    if len(shape) == 3: alpha = np.max(alpha, axis=0)
    if normal:
        image = np.asarray(image * 255)
    image = np.asarray(image, np.uint8)
    alpha = np.asarray(alpha * 255, np.uint8)
    masks = np.asarray(alpha > 200, np.uint8)
    if transpose:
        image = image_utils.untranspose(image)
    color_image, color_mask = color_utils.decode_color_image_mask(image, masks, data_type='coco')
    # color_image, color_mask = color_utils.decode_color_image_mask(image, mask, data_type='coco')
    fusion = image_utils.image_composite(image, alpha, bg_img=(67, 142, 219))
    vis = image_utils.image_hstack([image, alpha, color_image, fusion])
    # vis = image_utils.image_hstack([image, color_mask])
    # image_utils.cv_show_image("image", image)
    image_utils.cv_show_image("vis", vis, use_rgb=True)


if __name__ == "__main__":
    from pybaseutils.transforms import build_transform

    input_size = [416, 416]
    # class_name = ["布线"]
    class_name = ["布线通道", "胶枪头与布线通道有接触,胶枪头与布线通道无接触"]
    anno_dir = [
        "/home/PKing/Pictures/dataset-test/images",
    ]
    bg_dir = None
    transform = build_transform.image_transform(input_size,
                                                mean=[0, 0, 0],
                                                std=[1.0, 1.0, 1.0],
                                                padding=False,
                                                bg_dir=bg_dir,
                                                trans_type="test_regress",
                                                # trans_type="train_regress",
                                                )
    dataset = LabelmeLineDataset(filename=None,
                                 data_root=None,
                                 anno_dir=anno_dir,
                                 image_dir=None,
                                 # phase="test",
                                 class_name=class_name,
                                 transform=transform,
                                 check=False,
                                 resample=False,
                                 shuffle=False)
    print("have num:{}".format(len(dataset)))
    for i in range(len(dataset)):
        print(i)  #
        i = 0
        data = dataset.__getitem__(i)
        image, alpha = data["image"], data["target"]
        show_target_image(image, alpha, normal=True, transpose=True)
