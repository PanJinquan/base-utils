# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : 2025-04-01 13:42:03
# @Brief  :
# --------------------------------------------------------
"""
import os
from pybaseutils import file_utils, image_utils
import dlib
import cv2


class FaceDetector(object):
    def __init__(self, model_file):
        """
        :param model_file: 检测模型文件，(https://dlib.net/files/mmod_human_face_detector.dat.bz2)直接下载，然后解压
        """
        self.model = dlib.cnn_face_detection_model_v1(model_file)

    def detect(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 读取图像并转换为 RGB
        res = self.model(rgb, 1)
        boxes = []
        for face in res:
            x, y, w, h = face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height()
            boxes.append([x, y, x + w, y + h])
        return boxes

    def image_dir_detect(self, image_dir, vis=True, shuffle=False):
        """
        :param image_dir: list,*.txt ,image path or directory
        :return:
        """
        image_list = file_utils.get_files_lists(image_dir, shuffle=shuffle)
        for path in image_list:
            image = image_utils.read_image(path)
            boxes = self.detect(image)
            self.draw_result(image, boxes, vis=vis)

    def apply_mosaic(self, image, boxes):
        """对人脸框进行马赛克处理"""
        image = image_utils.apply_mosaic(image, boxes, radius=8, scale=[1.0, 1.0])
        return image

    def draw_result(self, image, boxes, vis=True):
        """
        :param image:
        :param boxes:
        :param vis:
        :return:
        """
        image = self.apply_mosaic(image, boxes)  # 马赛克处理
        image = image_utils.draw_image_boxes(image, boxes, color=(0, 255, 0), thickness=2)
        if vis: image_utils.cv_show_image("image", image)
        return image

@profile
def demo():
    # 下载预训练模型文件（需手动下载）
    # 下载地址：https://dlib.net/files/mmod_human_face_detector.dat.bz2,然后解压
    model_file = "/home/PKing/Downloads/mmod_human_face_detector.dat"
    image_dir = "/media/PKing/新加卷/SDK/base-utils/data/person"
    det = FaceDetector(model_file=model_file)
    det.image_dir_detect(image_dir)


if __name__ == '__main__':
    demo()
