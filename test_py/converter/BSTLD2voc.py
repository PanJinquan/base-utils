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
from pybaseutils.maker import maker_voc
from pybaseutils import file_utils, image_utils, yaml_utils


def get_objects_info(annotation):
    """
    [('name', 'O'), ('height', 'O'), ('width', 'O'), ('vehicles', 'O'), ('nVehicles', 'O')]
    解析标注信息
    """
    name: str = annotation['path']
    # name = os.path.basename(path)
    bboxes, labels = [], []
    for data in annotation['boxes']:
        box = [data['x_min'], data['y_min'], data['x_max'], data['y_max']]
        label = data['label']
        occluded = data['occluded']
        bboxes.append(box)
        labels.append(label)
    return name, bboxes, labels


def converter_BSTLD2voc(image_dir, annot_file, out_voc, vis=True):
    """
    将Bosch Small Traffic Lights Dataset(BSTLD)转换为VOC数据格式(xmin,ymin,xmax,ymax)
    :param image_dir: BITVehicle数据集图片(*.jpg)根目录
    :param annot_file: BITVehicle数据集标注文件VehicleInfo.mat
    :param out_voc: 输出VOC格式数据集目录
    :param vis: 是否可视化效果
    """
    phase = os.path.basename(annot_file).split(".")[0]
    print("image_dir :{}".format(image_dir))
    print("annot_file:{}".format(annot_file))
    print("out_voc   :{}".format(out_voc))
    # image_dir = os.path.join(image_dir, phase)
    out_image_dir = file_utils.create_dir(out_voc, None, "JPEGImages")
    out_xml_dir = file_utils.create_dir(out_voc, None, "Annotations")
    class_set = []
    annotations = yaml_utils.load_config(annot_file)
    for i in tqdm(range(len(annotations))):
        # i = 52
        image_name, bboxes, labels = get_objects_info(annotations[i])
        print("i={},labels:{}".format(i, labels))
        if len(labels) == 0:
            continue
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
        objects = maker_voc.create_objects(bboxes, labels)
        maker_voc.write_voc_xml_objects(new_name, image_shape, objects, xml_path)
        dst_file = file_utils.create_dir(out_image_dir, None, new_name)
        file_utils.copy_file(image_file, dst_file)
        # cv2.imwrite(dst_file, image)
        if vis:
            image = image_utils.draw_image_bboxes_text(image, bboxes, labels,
                                                       color=(255, 0, 0), thickness=1, fontScale=0.5)
            image_utils.cv_show_image("det", image, use_rgb=False, delay=0)
    file_utils.save_file_list(out_image_dir, filename=None, prefix="", postfix=file_utils.IMG_POSTFIX,
                              only_id=False, shuffle=False, max_num=None)
    print("class_set:{}".format(class_set))


if __name__ == "__main__":
    """
    pip install pybaseutils
    """
    image_dir = "/home/dm/nasdata/dataset/csdn/traffic light/Bosch Small Traffic Lights Dataset/dataset_train_rgb/"
    annot_file = "/home/dm/nasdata/dataset/csdn/traffic light/Bosch Small Traffic Lights Dataset/dataset_train_rgb/train.yaml"
    # 将车辆检测数据集BIT-Vehicle Dataset转换为VOC数据格式
    out_voc = os.path.join(os.path.dirname(image_dir), "VOC")
    converter_BSTLD2voc(image_dir, annot_file, out_voc, vis=True)
