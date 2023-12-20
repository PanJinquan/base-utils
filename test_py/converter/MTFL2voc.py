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
    name = annotation[1].replace("\\", "/")
    landmark = np.asarray(annotation[2:12])
    landmark = [landmark.reshape(2, 5).T]
    bboxes = image_utils.polygons2boxes(landmark)
    bboxes = image_utils.extend_xyxy(bboxes, scale=(2.0, 2.1))
    labels = ["face"]
    return name, bboxes, labels, landmark


def converter_BITVehicle2voc(annot_file, out_voc="", vis=True):
    """
    :param annot_file: BITVehicle数据集标注文件VehicleInfo.mat
    :param out_voc: 输出VOC格式数据集目录
    :param vis: 是否可视化效果
    """
    data_root = os.path.dirname(annot_file)
    if not out_voc: out_voc = os.path.join(data_root, "VOC")
    print("annot_file:{}".format(annot_file))
    print("out_voc   :{}".format(out_voc))
    out_image_dir = file_utils.create_dir(out_voc, None, "JPEGImages")
    out_xml_dir = file_utils.create_dir(out_voc, None, "Annotations")
    class_set = []
    print(annot_file)
    annotations = file_utils.read_data(annot_file, split=" ")
    for i in tqdm(range(len(annotations))):
        data = annotations[i]
        if len(data) != 16:
            print(data)
            continue
        image_path, bboxes, labels, landmark = parser_annotations(data)
        image_name = os.path.basename(image_path)
        image_id = image_name.split(".")[0]
        image_file = os.path.join(data_root, image_path)
        class_set = labels + class_set
        class_set = list(set(class_set))
        if not os.path.exists(image_file):
            print("not exist:{}".format(image_file))
            continue
        image = cv2.imread(image_file)
        image_shape = image.shape
        xml_path = file_utils.create_dir(out_xml_dir, None, "{}.xml".format(image_id))
        # objects = build_voc.create_objects(bboxes, labels)
        # build_voc.write_voc_xml_objects(image_name, image_shape, objects, xml_path)
        build_voc.write_voc_landm_xml_file(image_name, image_shape, bboxes, labels, landmark, xml_path)
        dst_file = file_utils.create_dir(out_image_dir, None, image_name)
        file_utils.copy_file(image_file, dst_file)
        # cv2.imwrite(dst_file, image)
        if vis:
            image = image_utils.draw_landmark(image, landmark)
            image = image_utils.draw_image_bboxes_text(image, bboxes, labels,
                                                       color=(255, 0, 0), thickness=2, fontScale=1.0)
            image_utils.cv_show_image("det", image, use_rgb=False, delay=0)
    file_utils.save_file_list(out_image_dir, filename=None, prefix="", postfix=file_utils.IMG_POSTFIX,
                              only_id=False, shuffle=False, max_num=None)
    print("class_set:{}".format(class_set))


if __name__ == "__main__":
    """
    pip install pybaseutils
    """
    annot_file = "/home/PKing/Downloads/MTFL/training.txt"
    # 将车辆检测数据集BIT-Vehicle Dataset转换为VOC数据格式
    converter_BITVehicle2voc(annot_file, vis=False)
