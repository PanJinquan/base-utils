# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-11-29 18:11:47
    @Brief  :
"""
import os
import numpy as np
from pybaseutils import image_utils, json_utils


def parser_dataset(data_root, anno_file, vis=True):
    """
    :param data_root: 数据集根目录
    :param anno_file: 标注文件
    :param vis: 可视化效果
    :return:
    """
    annotations: dict = json_utils.read_json_data(anno_file)
    for image_name, anno in annotations.items():
        image_name = image_name.replace("./", "")
        image_file = os.path.join(data_root, image_name)
        driver_state = anno['driver_state']
        landmarks = anno['landmarks']
        print("file={},label={}:".format(image_file, driver_state))
        if vis:
            landm = np.asarray([landmarks])
            image = image_utils.read_image(image_file)
            point = (10, 50)
            image = image_utils.draw_text(image, point, driver_state, thickness=2, fontScale=1.2, drawType="simple")
            image = image_utils.draw_landmark(image, landm)
            image_utils.cv_show_image("image", image)
    return annotations


if __name__ == "__main__":
    data_root = "./"
    anno_file = "annotations_holdout.json"
    parser_dataset(data_root, anno_file)
