# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-10-12 16:21:55
    @Brief  :
"""
import sys
import os

sys.path.insert(0, os.getcwd())
import cv2
import numpy as np
import traceback
from tqdm import tqdm
from pybaseutils.converter.convert_labelme2coco import Labelme2COCO
from pybaseutils.dataloader import parser_coco_kps
from pybaseutils import file_utils, image_utils, json_utils
from pybaseutils.converter import build_coco

skeleton = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
            [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
            [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]]


class HandPose(build_coco.COCOBuilder):
    def __init__(self, init_id=None):
        super(HandPose, self).__init__(init_id=init_id)

    @staticmethod
    def get_hand_pose(image, pts: dict, box):
        h, w = image.shape[:2]
        keypoints = np.zeros(shape=(len(pts), 2), dtype=np.float32)
        for k, v in pts.items():
            keypoints[int(k), :] = (v["x"], v["y"])
        boxes = image_utils.polygons2boxes([keypoints])
        box = image_utils.extend_xyxy(boxes, scale=[1.1, 1.1], valid_range=[0, 0, w, h])[0]
        return box, keypoints

    def handpose_datasets_v1(self, image_dir, json_dir, output="", num_joints=21, vis=False):
        if not output: output = os.path.join(os.path.dirname(image_dir), "COCO")
        save_file = os.path.join(output, "coco_data.json")
        save_dirs = os.path.join(output, "images")
        file_utils.create_dir(save_dirs)
        image_list = file_utils.get_images_list(image_dir)
        for image_file in tqdm(image_list):
            name = os.path.basename(image_file).split(".")[0]
            json_file = os.path.join(json_dir, f"{name}.json")
            if not os.path.exists(json_file):
                print("image_file = {}".format(image_file))
                print("not exists : {}".format(json_file))
            try:
                image = image_utils.read_image(image_file)
                anns = json_utils.read_json_data(json_file)['info']
                height, width = image.shape[:2]
                labels, boxes, contours, keypoints = [], [], [], []
                for ann in anns:
                    box, pts = self.get_hand_pose(image, ann['pts'], ann['bbox'])
                    pts_ = self.get_keypoints_info(pts, width, height, num_joints=num_joints)
                    boxes.append(box)
                    labels.append("hand")
                    keypoints.append(pts_)
                    if vis:
                        parser_coco_kps.show_target_image(image, [pts], [box], skeleton=skeleton)
                filename = os.path.basename(image_file)
                info = {"boxes": boxes, "labels": labels, "contours": contours, "keypoints": keypoints}
                self.addObjects(filename, info, width, height, num_joints=num_joints)
                file_utils.copy_file_to_dir(image_file, save_dirs)
            except:
                traceback.print_exc()
                print("Error = {}".format(image_file))
                print("Error : {}".format(json_file))
        # 设置关键点的名称和skeleton
        kps_name = [str(i) for i in list(range(num_joints))]
        self.set_keypoints_category(kps_name=kps_name, skeleton=skeleton, cat_id=0)
        build_coco.COCOTools.check_coco(self.coco)
        self.save_coco(save_file)

    def get_keypoints_info(self, keypoints: dict, width, height, num_joints):
        """
        keypoints=num_joints*3,x,y,visibility
        keypoints关节点的格式 : [x_1, y_1, v_1,...,x_k, y_k, v_k]
        其中x,y为Keypoint的坐标，v为可见标志
            v = 0 : 未标注点
            v = 1 : 标注了但是图像中不可见（例如遮挡）
            v = 2 : 标注了并图像可见
        实际预测时，不要求预测每个关节点的可见性
        :param keypoints:
        :param num_joints: 关键点个数
        :param width: 图像宽度
        :param height: 图像长度
        :return:
        """
        if len(keypoints) == 0: return []
        kps = np.zeros(shape=(num_joints, 3), dtype=np.int32) + 2
        kps[:, 0:2] = keypoints
        kps[:, 0] = np.clip(kps[:, 0], 0, width - 1)
        kps[:, 1] = np.clip(kps[:, 1], 0, height - 1)
        kps = kps.reshape(-1).tolist()
        return kps


if __name__ == '__main__':
    # image_dir = "/home/PKing/nasdata/dataset/tmp/hand-pose/handpose_datasets_v1/sample"
    image_dir = "/home/PKing/nasdata/dataset/tmp/hand-pose/handpose_datasets_v1/handpose_datasets_v1"
    h = HandPose()
    h.handpose_datasets_v1(image_dir, json_dir=image_dir, vis=False)
