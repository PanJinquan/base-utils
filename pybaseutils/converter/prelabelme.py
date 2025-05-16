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
from pybaseutils.converter import build_labelme
from pybaseutils import http_utils, base64_utils, file_utils, image_utils


class PreLabelme(object):
    """
    调用API接口进行预标注，并保持为Labelme数据格式
    接口必须返回字段:
            boxes 必选
            label 必选
            score 可选
            names 可选
            segms 可选
    """

    def __init__(self, url, headers=None, timeout=6, max_retries=1):
        """
        :param url:
        """
        self.url = url
        self.headers = headers
        self.timeout = timeout
        self.max_retries = max_retries

    def request(self, params, **kwargs):
        """
        :param params:
        :return: boxes,label,names,segms,,score
        """
        params.update(kwargs)
        params = base64_utils.serialization(params)
        result = http_utils.post(self.url, params=params, headers=self.headers,
                                 timeout=self.timeout, max_retries=self.max_retries)
        result = base64_utils.deserialization(result)
        return result

    def draw_result(self, image, result: dict, vis=True, delay=0):
        """
        :param image: BGR image
        :param boxes:<np.ndarray>: (num_boxes, 4), box=[xmin,ymin,xmax,ymax]
        :param score:<np.ndarray>: (num_boxes,)
        :param label:<np.ndarray>: (num_boxes,)
        :return:
        """
        color = (255, 0, 0)
        boxes = result.get("boxes", [])
        score = result.get("score", [])
        landm = result.get("landm", [])
        segms = result.get("segms", [])
        label = result.get("label", [])
        names = result.get("names", [])
        if not names: names = label
        texts = ["{} {:3.3f}".format(n, s) for n, s in zip(names, score)]
        image = image_utils.draw_image_bboxes_text(image, boxes, texts, color=color, thickness=2,
                                                   fontScale=0.8, drawType="ch")
        image = image_utils.draw_landmark(image, landm, radius=2, fontScale=1.0, color=color)
        if vis: image_utils.cv_show_image("image", image, delay=delay)
        return image

    def detect_image_dir(self, image_dir, vis=True):
        """
        :param image_dir: directory or image file path
        :param vis:<bool>
        :return:
        """
        image_list = file_utils.get_files_lists(image_dir)
        for image_file in tqdm(image_list):
            image = cv2.imread(image_file)
            result = self.request(params={"image": image})
            if vis: image = self.draw_result(image, result)

    def prelabel_labelme(self, image_dir, clsss_dict={}, vis=True):
        """
        进行预标注,并输出labelme数据格式
        :param image_dir: directory or image file path
        :param vis:<bool>
        :return:
        """
        image_list = file_utils.get_files_lists(image_dir)
        for image_file in tqdm(image_list):
            image = cv2.imread(image_file)
            if image is None: continue
            h, w = image.shape[:2]
            result = self.request(params={"image": image})
            image_name = os.path.basename(image_file)
            json_file = file_utils.change_postfix(image_file, postfix=".json")
            boxes = result.get("boxes", [])
            segms = result.get("segms", [])
            label = result.get("label", [])
            names = [clsss_dict[l] for l in label] if clsss_dict else result.get("names", [])
            result['names'] = names
            points = image_utils.boxes2polygons(boxes)
            build_labelme.maker_labelme(json_file, points, names, image_name, image_size=(w, h), image_bs64=None)
            if vis: image = self.draw_result(image, result)


if __name__ == '__main__':
    # url = 'http://192.168.68.102:5000/detect'
    url = 'http://0.0.0.0:5000/detect'
    image_dir = "/media/PKing/新加卷1/个人文件/video/video/frame"
    s = PreLabelme(url=url)
    s.prelabel_labelme(image_dir=image_dir, vis=False)
    # s.detect_image_dir(image_dir=image_dir)
