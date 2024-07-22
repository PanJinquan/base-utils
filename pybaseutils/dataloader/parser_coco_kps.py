# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail :
    @Date   : 2023-08-10 10:18:32
    @Brief  :
"""
import os
import numpy as np
from pybaseutils import image_utils, file_utils, color_utils
from pybaseutils.pose import bones_utils
from pybaseutils.dataloader import base_coco
from pybaseutils.dataloader.base_coco import CocoDataset, ConcatDataset

# skeleton连接线，keypoint关键点名称，num_joints关键点个数
BONES = bones_utils.BONES


class CocoKeypoint(base_coco.CocoDataset):
    def __init__(self, anno_file, image_dir="", class_name=[], num_joints=-1, transform=None, target_transform=None,
                 use_rgb=True, shuffle=False, decode=True, **kwargs):
        """
        initialize COCO api for keypoint annotations
        :param anno_file:
        :param image_dir:
        :param class_name:
        :param num_joints:
        :param skeleton:
        :param kwargs:
        """
        super(CocoKeypoint, self).__init__(anno_file, image_dir=image_dir, class_name=class_name, transform=transform,
                                           target_transform=target_transform, use_rgb=use_rgb,
                                           shuffle=shuffle, decode=decode, **kwargs)
        if not class_name: class_name = [self.class_name[1]]
        if isinstance(class_name, dict): class_name = list(class_name.keys())
        self.kps_info = self.load_keypoints_info(target=class_name)
        self.num_joints = num_joints
        if class_name[0] in BONES:
            self.bones = BONES[class_name[0]]
        else:
            coco_skeleton = self.kps_info[0]['skeleton']  # 关键点连接线
            coco_skeleton = np.asarray(coco_skeleton)
            skeleton = coco_skeleton - np.min(coco_skeleton)
            skeleton = skeleton.tolist()
            num_joints = np.max(skeleton) + 1
            self.bones = {"skeleton": skeleton,  # 关键点连接线
                          "keypoint": self.kps_info[0].get('keypoints', []),  # 关键点名称
                          "num_joints": num_joints,  # 关键点个数
                          "colors": None,  # 关键点连接线颜色
                          "names": []
                          }

        self.skeleton = self.bones["skeleton"]
        self.num_joints = self.bones["num_joints"]
        self.colors = self.bones["colors"]
        # skeleton下标从0开始，coco_skeleton下标是从1开始的
        self.coco_skeleton = np.array(self.skeleton, dtype=np.int32) + 1
        self.set_skeleton_keypoints(self.kps_info[0]['id'], skeleton=self.coco_skeleton, keypoints=[])
        print("num_joints             :{}".format(self.num_joints))
        print("skeleton               :{}".format(self.skeleton))
        print("coco skeleton          :{}".format(self.coco_skeleton.tolist()))
        print("anno_file              :{}".format(anno_file))
        print("------" * 10)

    def __getitem__(self, index):
        """
        :param vis:
        :return: 
        """
        image_id = self.image_ids[index]
        anns_info, file_info = self.get_object_annotations(image_id)
        image, width, height, image_file = self.get_object_image(file_info)
        boxes, labels, keypoints = self.get_keypoint_info(anns_info, self.num_joints)
        data = {"image": image, "boxes": boxes, "labels": labels,
                "segs": [], "mask": [], "keypoints": keypoints, "target": [],
                "image_id": image_id, "annotations": anns_info, "file_info": file_info,
                "image_file": image_file, "size": [width, height], "class_name": self.class_name}
        return data


def CocoKeypoints(anno_file=None,
                  image_dir="",
                  class_name=None,
                  transform=None,
                  target_transform=None,
                  use_rgb=True,
                  shuffle=False,
                  decode=True,
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
    :param decode: 是否对segment进行解码， True:在mask显示分割信息,False：mask为0，无分割信息
    :return:
    """
    if not isinstance(anno_file, list) and os.path.isfile(anno_file):
        anno_file = [anno_file]
    datasets = []
    for file in anno_file:
        data = CocoKeypoint(anno_file=file,
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


def show_target_image(image, keypoints, boxes, skeleton, colors=None, thickness=2, vis_id=False, use_rgb=True):
    image = image_utils.draw_key_point_in_image(image,
                                                keypoints,
                                                pointline=skeleton,
                                                boxes=boxes,
                                                vis_id=vis_id,
                                                thickness=thickness,
                                                colors=colors)
    image_utils.cv_show_image("keypoints", image, delay=0, use_rgb=use_rgb)


if __name__ == "__main__":
    # class_name = ["person"]
    # coco_root = "/home/PKing/nasdata/dataset/face_person/COCO/"
    # image_dir = coco_root + 'val2017/images'
    # anno_file = coco_root + 'annotations/person_keypoints_val2017.json'

    image_dir = "/media/PKing/新加卷1/SDK/base-utils/data/coco/JPEGImages"
    anno_file = "/media/PKing/新加卷1/SDK/base-utils/data/coco/coco_kps.json"
    class_name = ["person"]

    # hand
    image_dir = ""
    # anno_file = "/home/PKing/nasdata/dataset/tmp/hand-pose/HandPose-v2/train/train_anno.json"
    # anno_file = "/home/PKing/nasdata/dataset/tmp/hand-pose/HandPose-v1/test/test_anno.json"
    anno_file = "/home/PKing/nasdata/dataset/tmp/hand-pose/HandPose-v2/train/train_anno.json"
    class_name = []
    dataset = CocoKeypoint(anno_file, image_dir, class_name=class_name)
    skeleton = dataset.skeleton
    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        # data = {"segs": segs, "image": image, "boxes": boxes, "label": labels, "image_id": image_id}
        image, boxes, labels, keypoints = data['image'], data["boxes"], data["labels"], data["keypoints"]
        print("i={},image_id={}".format(i, data["image_id"]))
        # dataset.showAnns(image, data['annotations'])
        show_target_image(image, keypoints, boxes, skeleton=skeleton)
