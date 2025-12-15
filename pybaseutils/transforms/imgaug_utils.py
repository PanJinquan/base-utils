# -*- coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail :
    @Date   : 2024-07-25 08:33:57
    @Brief  :  https://github.com/aleju/imgaug-doc
               https://imgaug.readthedocs.io/en/latest/index.html
"""
import os
import PIL
import cv2
import numpy as np
from typing import List, Tuple
import imgaug as ia
import imgaug.augmenters.meta as meta
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from pybaseutils import file_utils, image_utils
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.kps import KeypointsOnImage
from imgaug.augmentables import Keypoint, KeypointsOnImage
from imgaug.augmentables.polys import Polygon, PolygonsOnImage


class Compose(object):
    def __init__(self, transforms: List = [], fixed=True):
        """
        :param transforms:
        :param fixed: images的图片是否使用相同的增强方式
        """
        self.fixed = fixed
        self.transforms = transforms if transforms else []
        self.aug = iaa.Sequential(children=self.transforms, random_order=False, seed=2004)

    def __call__(self, image=None, images=None, **kwargs):
        """
        kwargs = ["image","images", "heatmaps", "segmentation_maps",
                "keypoints", "bounding_boxes", "polygons",
                "line_strings"]
        :param kwargs:
        :return:
        """
        if isinstance(images, list) and len(images) > 0:
            if self.fixed:
                aug = self.aug.to_deterministic()
                images = [aug(image=images[i], **kwargs) for i in range(len(images))]
            else:
                images = self.aug(images=images, **kwargs)
            return images
        if isinstance(image, np.ndarray):
            image = self.aug(image=image, **kwargs)
            return image


class Transpose(meta.Augmenter):
    def __init__(self, p=1, seed=None, name=None, random_state="deprecated", deterministic="deprecated"):
        super(Transpose, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)
        self.p = iap.handle_probability_param(p, "p")

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.p]

    def _augment_images(self, images, random_state, parents, hooks):
        images = [self.task(img) for img in images]
        return images

    def task(self, image, **kwargs):
        image = image.transpose(2, 0, 1)
        return image


class ColorJitter(meta.Augmenter):
    """
    imgaug_utils.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.1, hue=0.1),
    """
    from torchvision.transforms import transforms
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5,
                 p=1, seed=None, name=None, random_state="deprecated", deterministic="deprecated"):
        super(ColorJitter, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)
        self.p = iap.handle_probability_param(p, "p")
        self.t = self.transforms.ColorJitter(brightness=brightness, contrast=contrast,
                                             saturation=saturation, hue=hue)

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.p]

    def _augment_images(self, images, random_state, parents, hooks):
        images = [self.task(img) for img in images]
        return images

    def task(self, image, **kwargs):
        """
        kwargs = ["image","images", "heatmaps", "segmentation_maps",
                "keypoints", "bounding_boxes", "polygons",
                "line_strings"]
        :param image:
        :param kwargs:
        :return:
        """
        image = PIL.Image.fromarray(image)
        image = self.t(image)
        image = np.asarray(image)
        return image


class Normalize(meta.Augmenter):
    def __init__(self, mean, std, p=1, seed=None, name=None, random_state="deprecated", deterministic="deprecated"):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        super(Normalize, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)
        self.p = iap.handle_probability_param(p, "p")

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.p]

    def _augment_images(self, images, random_state, parents, hooks):
        images = [self.task(img) for img in images]
        return images

    def task(self, image, **kwargs):
        """
        kwargs = ["image","images", "heatmaps", "segmentation_maps",
                "keypoints", "bounding_boxes", "polygons",
                "line_strings"]
        :param image:
        :param kwargs:
        :return:
        """
        image = np.asarray(image, dtype=np.float32) / 255.0
        image = (image - self.mean) / self.std
        return image


class ToTensor(meta.Augmenter):
    import torch
    def __init__(self, p=1, seed=None, name=None, random_state="deprecated", deterministic="deprecated"):
        super(ToTensor, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)
        self.p = iap.handle_probability_param(p, "p")

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.p]

    def _augment_images(self, images, random_state, parents, hooks):
        images = [self.task(img) for img in images]
        return images

    def task(self, image, **kwargs):
        """
        kwargs = ["image","images", "heatmaps", "segmentation_maps",
                "keypoints", "bounding_boxes", "polygons",
                "line_strings"]
        :param image:
        :param kwargs:
        :return:
        """
        image = self.torch.from_numpy(image).permute(2, 0, 1)
        return image


def decode_polygons(polygons: List, shape) -> ia.PolygonsOnImage:
    """
    :param polygons: [(N,2),(N,2),....,]
    :param shape: image.shape
    :return:
    """
    polygons = [Polygon(c) for c in polygons]
    P = ia.PolygonsOnImage(polygons, shape=shape)
    return P


def encode_polygons(polygons: ia.PolygonsOnImage) -> List:
    polygons = [item.coords for item in polygons.polygons]
    return polygons


aug_seq = iaa.Sequential([
    iaa.AddToHueAndSaturation(value_hue=(-25, 25), value_saturation=(-25, 25)),  # 饱和度 + 色调（hue）
    iaa.Multiply((0.5, 1.5)),  # 亮度：等效于 brightness ∈ [0.8, 1.2] → 用 Multiply
    iaa.LinearContrast((0.5, 1.5)),  # 对比度：contrast ∈ [0.8, 1.2]
    iaa.MotionBlur(k=15),  # 运动模糊
    iaa.Clouds(),  # 云雾
    iaa.imgcorruptlike.Fog(severity=1),  # 多雾/霜
    iaa.imgcorruptlike.Snow(severity=2),  # 下雨、大雪
    iaa.Rain(drop_size=(0.10, 0.20), speed=(0.2, 0.3)),  # 雨
    iaa.Rain(speed=(0.3, 0.5)),  # 雨
    iaa.Snowflakes(flake_size=(0.6, 0.7), speed=(0.02, 0.03)),  # 雪点
    iaa.imgcorruptlike.Spatter(severity=2),  # 溅 123水滴、45泥
    iaa.contrast.LinearContrast((0.5, 2.0), per_channel=0.5),  # 对比度变为原来的一半或者二倍
    iaa.imgcorruptlike.Brightness(severity=2),  # 亮度增加
    iaa.imgcorruptlike.Saturate(severity=3),  # 色彩饱和度
    iaa.FastSnowyLandscape(lightness_threshold=(100, 255), lightness_multiplier=(1.5, 2.0)),
    # 雪地亮度阈值是从 uniform(100, 255)（每张图像）和来自 uniform(1.5, 2.0)（每张图像）的乘数采样的。
    iaa.Cartoon(blur_ksize=3, segmentation_size=1.0, saturation=2.0, edge_prevalence=1.0),  # 卡通
])


def augment_example(input_size=(224, 224)):
    transforms = [
        iaa.Resize({"width": int(input_size[0] * 1.2), "height": "keep-aspect-ratio"}),
        iaa.Fliplr(0.5),  # 以75%的概率水平翻转图像
        iaa.AddToHueAndSaturation(value_hue=(-25, 25), value_saturation=(-25, 25)),  # 饱和度 + 色调（hue）
        iaa.Multiply((0.5, 1.5)),  # 亮度：等效于 brightness ∈ [0.8, 1.2] → 用 Multiply
        iaa.LinearContrast((0.5, 1.5)),  # 对比度：contrast ∈ [0.8, 1.2]
        iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                   translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                   rotate=(-5, 5),
                   shear=(-8, 8)
                   ),
        iaa.Crop(percent=(0, 0.1)),
        iaa.Resize({"width": input_size[0], "height": input_size[1]}),
        # 以50%的概率对图像进行小的高斯模糊增强
        # iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
        #
        # # 添加高斯噪声
        # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # # 使部分图像变亮，部分变暗
        # # 对图像进行仿射变换
        # iaa.PadToFixedSize(width=100, height=100),
        # iaa.CropToFixedSize(width=100, height=100),
        # iaa.CropToFixedSize(width=size[0], height=size[1]),
        # iaa.Resize({"width": int(size[0]), "height": "keep-aspect-ratio"}),
        # iaa.PadToFixedSize(width=size[0], height=size[1]),
        # iaa.CenterPadToFixedSize(width=size[0], height=size[1]),
    ]
    return transforms


def demo_for_image():
    # image = cv2.rotate(image, rotateCode=cv2.ROTATE_90_CLOCKWISE)
    transforms = augment_example()
    augment = Compose(transforms=transforms, fixed=True)
    for i in range(100):
        image_path = "../../data/test.png"
        image = image_utils.read_image(image_path)
        images = [image.copy(), image.copy()]
        images = augment(images=images)
        [print(img.shape) for img in images]
        result = image_utils.image_hstack(images)
        image_utils.cv_show_image("image", image, delay=5)
        image_utils.cv_show_image("result", result)


def demo_for_segment():
    transforms = augment_example()
    augment = Compose(transforms=transforms, fixed=True)
    for i in range(100):
        image = image_utils.read_image("../../data/test.png")
        mask = image_utils.read_image("../../data/mask.png")
        mask = image_utils.get_image_mask(mask)
        auimg, segimg = augment(image=image, segmentation_maps=SegmentationMapsOnImage(mask, shape=mask.shape))
        mask = np.asarray(segimg.arr[:, :, 0], dtype=np.uint8)
        color_image, color_mask = image_utils.draw_image_mask_color(auimg, mask)
        result = image_utils.image_hstack([image, color_image, mask])
        image_utils.cv_show_image("image", result)


def demo_for_keypoint():
    transforms = augment_example()
    augment = Compose(transforms=transforms, fixed=True)
    for i in range(100):
        image = image_utils.read_image("../../data/test.png")
        mask = image_utils.read_image("../../data/mask.png")
        mask = image_utils.get_image_mask(mask)
        contours = image_utils.find_mask_contours(mask)
        auimg, contours = augment(image=image, keypoints=contours)
        h, w = auimg.shape[:2]
        mask = image_utils.draw_mask_contours(contours, size=(w, h))
        color_image = image_utils.draw_image_contours(auimg, contours)
        result = image_utils.image_hstack([image, color_image, mask])
        image_utils.cv_show_image("image", result)


def demo_for_polygons():
    transforms = augment_example()
    augment = Compose(transforms=transforms, fixed=True)
    for i in range(100):
        image = image_utils.read_image("../../data/test.png")
        mask = image_utils.read_image("../../data/mask.png")
        mask = image_utils.get_image_mask(mask)
        polygons = image_utils.find_mask_contours(mask)  # List[np.ndarray(num_point,2)]
        # contours = contours * 3
        polygons.append(np.asarray([[50, 50], [50, 100], [100, 100]], dtype=np.int32))
        # polygons = decode_polygons(polygons, image.shape)
        auimg, polygons = augment(image=image, polygons=polygons)
        # polygons = encode_polygons(polygons)
        h, w = auimg.shape[:2]
        mask = image_utils.draw_mask_contours(polygons, size=(w, h))
        color_image = image_utils.draw_image_contours(auimg, polygons)
        result = image_utils.image_hstack([image, color_image, mask])
        image_utils.cv_show_image("image", result)


if __name__ == "__main__":
    # demo_for_image()
    # demo_for_segment()
    # demo_for_keypoint()
    demo_for_polygons()
