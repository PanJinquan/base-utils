# -*-coding: utf-8 -*-
"""
    @Project: PythonAPI
    @File   : cocoDemo.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-05-07 16:33:01
"""
import os
import numpy as np
from pybaseutils import image_utils, file_utils, color_utils
from pybaseutils.dataloader import base_coco


class CocoInstances(base_coco.CocoDataset):
    def __init__(self, anno_file, image_dir="", class_name=[], transform=None,
                 target_transform=None, use_rgb=False,
                 shuffle=False, check=False, **kwargs):
        """
        initialize COCO api for instance annotations
        :param anno_file:
        :param image_dir:
        :param class_name:
        :param transform:
        :param target_transform:
        :param use_rgb:
        :param shuffle:
        :param check:
        :param kwargs:
        """
        super(CocoInstances, self).__init__(anno_file, image_dir=image_dir, class_name=class_name, transform=transform,
                                            target_transform=target_transform, use_rgb=use_rgb,
                                            shuffle=shuffle, check=check, **kwargs)

    def __getitem__(self, index):
        """
        :param vis:
        :return: 
        """
        image_id = self.image_id[index]
        anns_info, file_info = self.get_object_annotations(image_id)
        image, width, height = self.get_object_image(file_info)
        boxes, labels, mask, segs = self.get_object_instance(anns_info, h=height, w=width)
        data = {"segs": segs, "mask": mask, "image": image, "boxes": boxes, "label": labels, "image_id": image_id,
                "annotations": anns_info, "file_info": file_info}
        return data


def show_target_image(image, mask, boxes, labels, class_name=[], thickness=2, fontScale=1.0):
    mask = np.asarray(mask, np.uint8)
    color_image, color_mask = color_utils.decode_color_image_mask(image, mask)
    color_image = image_utils.draw_image_bboxes_labels(color_image, boxes, labels, class_name=class_name,
                                                       thickness=thickness, fontScale=fontScale)
    vis_image = image_utils.image_hstack([image, mask, color_image, color_mask])
    image_utils.cv_show_image("image", vis_image)


if __name__ == "__main__":
    class_name = []
    # 测试COCO数据集
    coco_root = "/home/PKing/nasdata/dataset/face_person/COCO/"
    image_dir = coco_root + 'val2017/images'
    # anno_file = coco_root + 'annotations/person_keypoints_val2017.json'
    anno_file = coco_root + 'annotations/instances_val2017.json'

    anno_file = "/media/PKing/新加卷1/SDK/base-utils/data/VOC2007/voc_coco_demo.json"
    image_dir = "/media/PKing/新加卷1/SDK/base-utils/data/VOC2007/JPEGImages"

    anno_file = "/media/PKing/新加卷1/SDK/base-utils/data/person.json"
    image_dir = "/media/PKing/新加卷1/SDK/base-utils/data/person"
    dataset = CocoInstances(anno_file, image_dir, class_name=class_name)
    class_name = dataset.class_name
    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        image, boxes, labels, mask = data['image'], data["boxes"], data["label"], data["mask"]
        print("i={},image_id={}".format(i, data["image_id"]))
        dataset.showAnns(image, data['annotations'])
        # show_target_image(image, mask, boxes, labels, class_name=class_name)
