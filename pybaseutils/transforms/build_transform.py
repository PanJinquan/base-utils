# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2025-12-03 14:17:27
# @Brief  :
# --------------------------------------------------------
"""
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from pybaseutils.transforms import imgaug_utils


def image_transform(input_size, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], padding=False, bg_dir=None,
                    trans_type="train"):
    """
    :param input_size: input_size=[W, H],size=[H,W]
    :param mean:
    :param std:
    :param trans_type:
    :return:
    """
    # TODO order = 插值方法（0=最近邻，1=双线性(默认)）
    # TODO ColorJitter(brightness=0.5, contrast=0.5, saturation=0.1, hue=0.1, trans_index=trans_index)
    if trans_type in ["train_regress", "train_reg"]:
        transforms = [
            iaa.Resize({"width": int(input_size[0] * 1.2), "height": "keep-aspect-ratio"}),
            iaa.Fliplr(0.5),  # 以50%的概率水平翻转图像
            # iaa.AddToHueAndSaturation(value_hue=(-25, 25), value_saturation=(-25, 25)),  # 饱和度 + 色调（hue）
            imgaug_utils.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.1, hue=0.1),
            iaa.Multiply((0.5, 1.5)),  # 亮度：等效于 brightness ∈ [0.8, 1.2] → 用 Multiply
            iaa.LinearContrast((0.5, 1.5)),  # 对比度：contrast ∈ [0.8, 1.2]
            iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                       translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                       rotate=(-10, 10),
                       shear=(-8, 8)
                       ),  # 仿射变换增强
            iaa.Crop(percent=(0, 0.3)),  # 随机裁剪
            iaa.Resize({"width": input_size[0], "height": input_size[1]}),
            imgaug_utils.Normalize(mean=mean, std=std),
        ]
    elif trans_type in ["test_regress", "test_reg"]:
        transforms = [
            iaa.Resize({"width": input_size[0], "height": input_size[1]}),
            imgaug_utils.Normalize(mean=mean, std=std),
        ]
    else:
        raise NotImplementedError("no {} transform implemented you have defined.".format(trans_type))
    transform = imgaug_utils.Compose(transforms=transforms, fixed=True)
    return transform
