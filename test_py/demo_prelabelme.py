# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : 2025-04-29 09:13:09
# @Brief  :
# --------------------------------------------------------
"""
import os
import cv2
from tqdm import tqdm
from pybaseutils.converter import build_labelme, prelabelme
from pybaseutils import http_utils, base64_utils, file_utils, image_utils


class PreLabelme(prelabelme.PreLabelme):

    def prelabel_labelme(self, image_dir, clsss_dict={}, vis=True):
        """
        进行预标注
        :param image_dir: directory or image file path
        :param vis:<bool>
        :return:
        """
        image_list = file_utils.get_files_lists(image_dir, shuffle=False)
        for image_file in tqdm(image_list):
            image = cv2.imread(image_file)
            if image is None: continue
            h, w = image.shape[:2]
            result = self.request(params={"image": image})
            image_name = os.path.basename(image_file)
            json_file = file_utils.change_postfix(image_file, postfix=".json")
            boxes = result.get("boxes", [])
            segms = result.get("segms", [])
            # label = result.get("label", [])
            if len(boxes) != 1: continue
            boxes = image_utils.clip_xyxy(boxes, valid_range=(0, 0, w, h))
            names = [os.path.basename(os.path.dirname(image_file))] * len(boxes)
            result['names'] = names
            points = image_utils.boxes2polygons(boxes)
            build_labelme.maker_labelme(json_file, points, names, image_name, image_size=(w, h), image_bs64=None)
            if vis: image = self.draw_result(image, result)


if __name__ == '__main__':
    # url = 'http://192.168.68.102:5000/detect'
    url = 'http://0.0.0.0:5000/detect'
    # image_dir = "/media/PKing/新加卷1/个人文件/video/video/frame"
    # image_dir = "/home/PKing/nasdata/tmp/tmp/RealFakeFace/living/portrait/videos/portrait2-frame"
    image_dir = "/home/PKing/nasdata/tmp/tmp/RealFakeFace/living/dataset-v1/image"
    # image_dir = "/home/PKing/nasdata/tmp/tmp/RealFakeFace/living/portrait/image/portrait220"
    # image_dir = "/home/PKing/nasdata/tmp/tmp/RealFakeFace/living/portrait/image/portrait764"
    clsss_dict = {}
    s = PreLabelme(url=url)
    s.prelabel_labelme(image_dir=image_dir, clsss_dict=clsss_dict, vis=False)
    # s.detect_image_dir(image_dir=image_dir)
