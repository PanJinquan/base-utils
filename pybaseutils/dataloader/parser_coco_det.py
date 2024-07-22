# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail :
    @Date   : 2023-08-10 10:18:32
    @Brief  :
"""
import os
import numpy as np
import cv2
import random
import json
from collections import defaultdict
from pybaseutils.dataloader.base_coco import CocoDataset, ConcatDataset
from pybaseutils import image_utils, file_utils


class CocoDetection(CocoDataset):
    """Coco dataset."""

    def __init__(self, anno_file, image_dir="", class_name=[], transform=None, target_transform=None, use_rgb=True,
                 shuffle=False, decode=False, **kwargs):
        """
        initialize COCO api for object detection annotations
        ├── annotations
        │    ├── instances_train2017.json
        │    └── person_keypoints_train2017.json
        └── images
        :param anno_file: COCO annotation file(*.json).
        :param image_dir: COCO image directory. 如果image_dir为空，则自动搜寻可能存在图片目录
        :param transform:(callable, optional): Optional transform to be applied on a sample.
        :param target_transform:
        :param use_rgb:
        :param shuffle:
        :param decode: 是否对segment进行解码， True:在mask显示分割信息,False：mask为0，无分割信息
        """
        super(CocoDetection, self).__init__(anno_file, image_dir=image_dir, class_name=class_name, transform=transform,
                                            target_transform=target_transform, use_rgb=use_rgb,
                                            shuffle=shuffle, decode=decode, **kwargs)
        print("CocoDataset class_name :{}".format(class_name))
        print("CocoDataset class_dict :{}".format(self.class_dict))
        print("CocoDataset num images :{}".format(len(self.image_ids)))
        print("CocoDataset num_classes:{}".format(self.num_classes))

    def convert_target(self, boxes, labels):
        # （xmin,ymin,xmax,ymax,label）
        if len(boxes) == 0:
            target = np.empty(shape=(0, 5), dtype=np.float32)
        else:
            target = np.concatenate([boxes, labels.reshape(-1, 1)], axis=1)
            target = np.asarray(target, dtype=np.float32)
        return target

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        anns_info, file_info = self.get_object_annotations(image_id)
        image, width, height, image_file = self.get_object_image(file_info)
        boxes, labels = self.get_object_detection(anns_info)
        if self.transform and len(boxes) > 0:
            image, boxes, labels = self.transform(image, boxes, labels)
        num_boxes = len(boxes)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        target = self.convert_target(boxes, labels)
        if num_boxes == 0:
            index = int(random.uniform(0, len(self)))
            return self.__getitem__(index)
        data = {"image": image, "boxes": boxes, "labels": labels,
                "segs": [], "mask": [], "keypoints": [], "target": target,
                "image_id": image_id, "annotations": anns_info, "file_info": file_info,
                "image_file": image_file, "size": [width, height], "class_name": self.class_name}
        return data


def CocoDetections(anno_file=None,
                   image_dir="",
                   class_name=None,
                   transform=None,
                   target_transform=None,
                   use_rgb=True,
                   shuffle=False,
                   decode=False,
                   **kwargs):
    """
    :param anno_file: str or List[str]
    :param data_root:
    :param json_dir:
    :param image_dir:
    :param class_name:
    :param transform:
    :param use_rgb:
    :param keep_difficult:
    :param shuffle:
    :param decode:
    :return:
    """
    # from torch.utils.data.dataset import ConcatDataset
    if not isinstance(anno_file, list) and os.path.isfile(anno_file):
        anno_file = [anno_file]
    datasets = []
    for file in anno_file:
        data = CocoDetection(anno_file=file,
                             image_dir=image_dir,
                             class_name=class_name,
                             transform=transform,
                             target_transform=target_transform,
                             use_rgb=use_rgb,
                             shuffle=shuffle,
                             decode=decode,
                             **kwargs)
        datasets.append(data)
    datasets = ConcatDataset(datasets, shuffle=shuffle)
    return datasets


def show_target_image(image, boxes, labels, normal=False, transpose=False, class_name=None,
                      thickness=2, fontScale=1.0, use_rgb=True):
    """
    :param image:
    :param targets_t:
                bboxes = targets[idx][:, :4].data
                keypoints = targets[idx][:, 4:14].data
                labels = targets[idx][:, -1].data
    :return:
    """
    import numpy as np
    from pybaseutils import image_utils
    image = np.asarray(image)
    boxes = np.asarray(boxes)
    labels = np.asarray(labels)
    # print("image:{}".format(image.shape))
    # print("bboxes:{}".format(bboxes))
    # print("labels:{}".format(labels))
    if transpose:
        image = image_utils.untranspose(image)
    h, w, _ = image.shape
    landms_scale = np.asarray([w, h] * 5)
    bboxes_scale = np.asarray([w, h] * 2)
    if normal:
        boxes = boxes * bboxes_scale
    image = image_utils.draw_image_bboxes_labels(image, boxes, labels, class_name=class_name,
                                                 thickness=thickness, fontScale=fontScale, drawType="chinese")
    image_utils.cv_show_image("image", image, delay=0, use_rgb=use_rgb)
    print("===" * 10)
    return image


if __name__ == "__main__":
    size = [640, 640]
    anno_file = "/home/PKing/nasdata/dataset/tmp/hand-pose/HandPose-v2/train/train_anno.json"
    dataset = CocoDetection(anno_file, class_name=[], transform=None, use_rgb=True)
    class_name = dataset.class_name
    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        image, targets, image_id = data['image'], data["target"], data["image_id"]
        bboxes, labels = targets[:, 0:4], targets[:, 4:5]
        print("i={},image_id={}".format(i, data["image_id"]))
        show_target_image(image, bboxes, labels, normal=False, transpose=False, class_name=class_name)
