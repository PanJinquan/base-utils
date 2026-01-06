# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : 2025-12-18 15:10:41
# @Brief  :
# --------------------------------------------------------
"""
import os
import cv2
import numpy as np
from pybaseutils import file_utils, image_utils
from PIL import Image
from pyzbar import pyzbar


class QRCodeDetector:
    def __init__(self, use_cv=True):
        self.use_cv = use_cv
        self.detector = cv2.QRCodeDetector()  # 创建 QRCodeDetector 对象

    @staticmethod
    def qrcode_pyzbar(image):
        """
        识别并解码指定图像中的二维码内容
        :param image: 二维码图片，支持 numpy.ndarray 和 PIL.Image 格式
        :return: 二维码中的文本内容，若未识别则返回 None
        """
        if isinstance(image, np.ndarray): image = Image.fromarray(image)
        results = pyzbar.decode(image)  # 解码二维码/条形码
        names, points = [], []
        for obj in results:
            t = obj.data.decode('utf-8')
            p = np.asarray(obj.polygon)
            names.append(t)
            points.append(p)
        return names, points

    def qrcode_opencv(self, image):
        """
        识别并解码指定图像中的二维码内容
        :param image: 二维码图片，支持 numpy.ndarray 和 PIL.Image 格式
        :return: 二维码中的文本内容，若未识别则返回 None
        """
        # name, point, qrimg = self.detector.detectAndDecode(image) # 单个二维码检测
        retval, names, points, qrimgs = self.detector.detectAndDecodeMulti(image)  # 识别多个二维码
        if not retval:
            names, points = [], []
        return names, points

    def task(self, image):
        if self.use_cv:
            names, points = self.qrcode_opencv(image)
        else:
            names, points = self.qrcode_pyzbar(image)
        return names, points

    def test_video_file(self, video_file, delay=10):
        """
        测试视频文件是否能识别二维码
        :param video_file: 视频文件路径
        :return:
        """
        cap = cv2.VideoCapture(video_file)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            names, points = self.task(frame)
            print(names)
            frame = image_utils.draw_image_contours(frame, points, names, color=(0, 0, 255), thickness=4, alpha=0.0)
            image_utils.show_image("frame", frame, delay=delay)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
        cv2.destroyAllWindows()

    def test_image_dir(self, image_dir):
        """
        测试目录下所有图片是否能识别二维码
        :param image_dir: 图片目录
        :return:
        """
        files = file_utils.get_files(image_dir)
        for image_path in files:
            image = image_utils.read_image(image_path, use_rgb=False)
            names, points = self.task(image)
            print(image_path, names)
            image = image_utils.draw_image_contours(image, points, names, color=(0, 0, 255), thickness=4, alpha=0.0)
            image_utils.show_image("image", image)


if __name__ == "__main__":
    """
    pip install pyzbar pybaseutils
    """
    detector = QRCodeDetector(use_cv=False)
    image_dir = "/home/PKing/Downloads/qrcode/data"
    detector.test_image_dir(image_dir)
    video_file = "/home/PKing/Downloads/qrcode/data/video.mp4"
    detector.test_video_file(video_file)
