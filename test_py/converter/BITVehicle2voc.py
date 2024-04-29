# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-11-29 18:49:56
    @Brief  :
"""

import os
import cv2
import scipy.io as scio
import numpy as np
from tqdm import tqdm
from pybaseutils.converter import build_voc
from pybaseutils import file_utils, image_utils


def parser_annotations(annotation):
    """
    [('name', 'O'), ('height', 'O'), ('width', 'O'), ('vehicles', 'O'), ('nVehicles', 'O')]
    解析标注信息
    """
    name = annotation["name"][0][0]
    height = annotation["height"][0][0][0]
    wideth = annotation["width"][0][0][0]
    nvehicles = annotation["nVehicles"][0][0][0]
    vehicles = annotation["vehicles"][0][0]
    bboxes, labels = [], []
    for i in range(nvehicles):
        box = [vehicles[n][i][0][0] for n in ['left', 'top', 'right', 'bottom']]
        label = vehicles['category'][i][0]
        bboxes.append(box)
        labels.append(label)
    return name, nvehicles, bboxes, labels


def converter_BITVehicle2voc(image_dir, annot_file, out_voc, vis=True):
    """
    将车辆检测数据集BITVehicle转换为VOC数据格式(xmin,ymin,xmax,ymax)
    :param image_dir: BITVehicle数据集图片(*.jpg)根目录
    :param annot_file: BITVehicle数据集标注文件VehicleInfo.mat
    :param out_voc: 输出VOC格式数据集目录
    :param vis: 是否可视化效果
    """
    print("image_dir :{}".format(image_dir))
    print("annot_file:{}".format(annot_file))
    print("out_voc   :{}".format(out_voc))
    out_image_dir = file_utils.create_dir(out_voc, None, "JPEGImages")
    out_xml_dir = file_utils.create_dir(out_voc, None, "Annotations")
    class_set = []
    print(annot_file)
    annotations = scio.loadmat(annot_file)['VehicleInfo']
    for i in tqdm(range(len(annotations))):
        # i = 52
        image_name, nvehicles, bboxes, labels = parser_annotations(annotations[i])
        print("i={},nvehicles:{},labels:{}".format(i, nvehicles, labels))
        image_id = image_name.split(".")[0]
        image_file = os.path.join(image_dir, image_name)
        class_set = labels + class_set
        class_set = list(set(class_set))
        if not os.path.exists(image_file):
            print("not exist:{}".format(image_file))
            continue
        image = cv2.imread(image_file)
        image_shape = image.shape
        new_name = "{}.jpg".format(image_id)
        xml_path = file_utils.create_dir(out_xml_dir, None, "{}.xml".format(image_id))
        objects = build_voc.create_objects(bboxes, labels)
        build_voc.write_voc_xml_objects(new_name, image_shape, objects, xml_path)
        dst_file = file_utils.create_dir(out_image_dir, None, new_name)
        file_utils.copy_file(image_file, dst_file)
        # cv2.imwrite(dst_file, image)
        if vis:
            image = image_utils.draw_image_bboxes_text(image, bboxes, labels,
                                                       color=(255, 0, 0), thickness=2, fontScale=1.0)
            image_utils.cv_show_image("det", image, use_rgb=False, delay=10)
    file_utils.save_file_list(out_image_dir, filename=None, prefix="", postfix=file_utils.IMG_POSTFIX,
                              only_id=False, shuffle=False, max_num=None)
    print("class_set:{}".format(class_set))


if __name__ == "__main__":
    """
    pip install pybaseutils
    """
    image_dir = "/home/dm/nasdata/dataset/csdn/car/BITVehicle/JPEGImages"
    annot_file = "/home/dm/nasdata/dataset/csdn/car/BITVehicle/VehicleInfo.mat"
    # 将车辆检测数据集BIT-Vehicle Dataset转换为VOC数据格式
    out_voc = os.path.join(os.path.dirname(image_dir), "VOC")
    converter_BITVehicle2voc(image_dir, annot_file, out_voc, vis=True)
