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
from pybaseutils.dataloader import base_coco


class CocoKeypoints(base_coco.CocoDataset):
    def __init__(self, anno_file, image_dir="", class_name=[], num_joints=-1, **kwargs):
        """
        initialize COCO api for keypoint annotations
        :param anno_file:
        :param image_dir:
        :param class_name:
        :param num_joints:
        :param skeleton:
        :param kwargs:
        """
        super(CocoKeypoints, self).__init__(anno_file, image_dir=image_dir, class_name=class_name, transform=None,
                                            target_transform=None, use_rgb=False,
                                            shuffle=False, check=False, **kwargs)
        self.kps_info = self.load_keypoints_info(target=class_name)
        self.image_dir = image_dir
        self.num_joints = num_joints
        if "coco_person" in class_name:
            self.skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                             [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]
            self.keypoints = self.kps_info[0]['keypoints']  # 关键点名称
        elif "pig" in class_name:
            self.skeleton = [[0, 1], [1, 2], [2, 3]]
        elif "finger" in class_name:
            self.skeleton = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 1), (10, 3), (10, 5), (10, 7), (10, 9)]
            self.keypoints = {"finger0": 0, "finger1": 1, "finger2": 2, "finger3": 3, "finger4": 4,
                              "finger5": 5, "finger6": 6, "finger7": 7, "finger8": 8, "finger9": 9,
                              "finger10": 10}
            # elif "finger_pen" in COCO_NAME:
            self.keypoints = {"finger0": 0, "finger1": 1, "finger2": 2, "finger3": 3, "finger4": 4,
                              "finger5": 5, "finger6": 6, "finger7": 7, "finger8": 8, "finger9": 9,
                              "finger10": 10, "pen0": 11, "pen1": 12}
            self.skeleton = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 1), (10, 3), (10, 5), (10, 7), (10, 9),
                             (11, 12)]
        elif "hand" in class_name:
            "http://challenge.xfyun.cn/topic/info?type=hand-key-point"
            self.skeleton = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [0, 6], [0, 7], [0, 8],
                             [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
                             [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]]
            self.num_joints = 21
        else:
            # self.keypoints = self.kps_info[0]['keypoints']  # 关键点名称
            coco_skeleton = self.kps_info[0]['skeleton']  # 关键点连接线
            coco_skeleton = np.asarray(coco_skeleton)
            self.skeleton = coco_skeleton - np.min(coco_skeleton)
            self.skeleton = self.skeleton.tolist()
            self.num_joints = np.max(self.skeleton) + 1  #
        # skeleton下标从0开始，coco_skeleton下标是从1开始的
        self.coco_skeleton = np.array(self.skeleton, dtype=np.int32) + 1
        self.set_skeleton_keypoints(self.kps_info[0]['id'], skeleton=self.coco_skeleton, keypoints=[])
        print("skeleton               :{}".format(self.skeleton))
        print("coco skeleton          :{}".format(self.coco_skeleton.tolist()))
        print("anno_file              :{}".format(anno_file))

    def __getitem__(self, index):
        """
        :param vis:
        :return: 
        """
        image_id = self.image_ids[index]
        anns_info, file_info = self.get_object_annotations(image_id)
        image, width, height = self.get_object_image(file_info)
        boxes, labels, keypoints = self.get_keypoint_info(anns_info, self.num_joints)
        data = {"keypoints": keypoints, "image": image, "boxes": boxes, "label": labels, "image_ids": image_id,
                "annotations": anns_info, "file_info": file_info}
        return data


def show_target_image(image, keypoints, boxes, labels, skeleton, vis_id=False):
    image = image_utils.draw_key_point_in_image(image,
                                                keypoints,
                                                pointline=skeleton,
                                                boxes=boxes,
                                                vis_id=vis_id,
                                                thickness=2)
    image_utils.cv_show_image("keypoints", image, delay=0)


if __name__ == "__main__":
    size = [640, 640]
    class_name = ["person"]
    coco_root = "/home/PKing/nasdata/dataset/face_person/COCO/"
    image_dir = coco_root + 'val2017/images'
    anno_file = coco_root + 'annotations/person_keypoints_val2017.json'

    # anno_file = "/media/PKing/新加卷1/SDK/base-utils/data/person.json"
    image_dir = "/media/PKing/新加卷1/SDK/base-utils/data/person"
    # image_dir = "/home/PKing/nasdata/dataset/tmp/challenge/精细化手部关键点检测挑战赛/精细化手部关键点检测挑战赛公开数据-初赛/训练集/image"
    # anno_file = "/home/PKing/nasdata/dataset/tmp/challenge/精细化手部关键点检测挑战赛/精细化手部关键点检测挑战赛公开数据-初赛/训练集/train_anno.json"
    class_name = ["hand"]
    dataset = CocoKeypoints(anno_file, image_dir, class_name=class_name)
    skeleton = dataset.skeleton
    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        # data = {"segs": segs, "image": image, "boxes": boxes, "label": labels, "image_ids": image_ids}
        image, boxes, labels, keypoints = data['image'], data["boxes"], data["label"], data["keypoints"]
        print("i={},image_ids={}".format(i, data["image_ids"]))
        # dataset.showAnns(image, data['annotations'])
        show_target_image(image, keypoints, boxes, labels, skeleton=skeleton)
