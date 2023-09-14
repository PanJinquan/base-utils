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

# skeleton连接线，keypoint关键点名称，num_joints关键点个数
BONES = {
    "coco_person": {
        # "skeleton": [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
        #              [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]],
        "skeleton": [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                     [6, 8], [7, 9], [8, 10], [0, 1], [0, 2], [1, 3], [2, 4]],
        "keypoint": [],
        "num_joints": 17,
        "class_dict": {0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear", 5: "left_shoulder",
                       6: "right_shoulder", 7: "left_elbow", 8: "right_elbow", 9: "left_wrist", 10: "right_wrist",
                       11: "left_hip", 12: "right_hip", 13: "left_knee", 14: "right_knee", 15: "left_ankle",
                       16: "right_ankle"}
    },
    "mpii": {
        "skeleton": [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
                     [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
                     [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]],
        "keypoint": [],
        "num_joints": 21,
        "class_dict": {0: "r_ankle", 1: "r_knee", 2: "r_hip", 3: "l_hip", 4: "l_knee", 5: "l_ankle", 6: "pelvis",
                       7: "thorax", 8: "upper_neck", 9: "head_top", 10: " r_wrist", 11: "r_elbow", 12: "r_shoulder",
                       13: "l_shoulder", 14: "l_elbow", 15: "l_wrist"}
    },
    "hand": {
        "skeleton": [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9],
                     [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
                     [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]],
        "keypoint": [],
        "num_joints": 21,

    },
}


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
                                            shuffle=False, decode=True, **kwargs)
        if not class_name: class_name = [self.class_name[1]]
        self.kps_info = self.load_keypoints_info(target=class_name)
        self.num_joints = num_joints
        if class_name[0] in BONES:
            self.skeleton = BONES[class_name[0]]["skeleton"]  # 关键点连接线
            self.keypoint = BONES[class_name[0]]["keypoint"]  # 关键点名称
            self.num_joints = BONES[class_name[0]]["num_joints"]
        else:
            self.keypoints = self.kps_info[0].get('keypoints', [])  # 关键点名称
            coco_skeleton = self.kps_info[0]['skeleton']  # 关键点连接线
            coco_skeleton = np.asarray(coco_skeleton)
            self.skeleton = coco_skeleton - np.min(coco_skeleton)
            self.skeleton = self.skeleton.tolist()
            self.num_joints = np.max(self.skeleton) + 1  #
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
        data = {"keypoints": keypoints, "image": image, "boxes": boxes, "label": labels, "image_ids": image_id,
                "annotations": anns_info, "file_info": file_info}
        return data


def show_target_image(image, keypoints, boxes, skeleton, vis_id=False):
    image = image_utils.draw_key_point_in_image(image,
                                                keypoints,
                                                pointline=skeleton,
                                                boxes=boxes,
                                                vis_id=vis_id,
                                                thickness=1)
    image_utils.cv_show_image("keypoints", image, delay=0)


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
    dataset = CocoKeypoints(anno_file, image_dir, class_name=class_name)
    skeleton = dataset.skeleton
    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        # data = {"segs": segs, "image": image, "boxes": boxes, "label": labels, "image_ids": image_ids}
        image, boxes, labels, keypoints = data['image'], data["boxes"], data["label"], data["keypoints"]
        print("i={},image_ids={}".format(i, data["image_ids"]))
        # dataset.showAnns(image, data['annotations'])
        show_target_image(image, keypoints, boxes, skeleton=skeleton)
