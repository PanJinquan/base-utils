# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-10-12 16:21:55
    @Brief  :
"""
import os
import numpy as np
import cv2
from tqdm import tqdm
from pybaseutils.dataloader import parser_voc
from pybaseutils.converter import build_voc, build_labelme
from pybaseutils import file_utils, image_utils, json_utils


def parser_annotations(xml_file):
    """
    解析CVAT标注工具
    :param xml_file:
    :return:
    """
    data = parser_voc.VOCDataset.read_xml2json(xml_file)
    height = int(data['annotation']['imagesize']['nrows'])
    width = int(data['annotation']['imagesize']['ncols'])
    filename = data['annotation']['filename']
    annos = []
    data_obj = json_utils.get_value(data, ['annotation', 'object'], default=None)
    if data_obj is None: return filename, annos, width, height
    if isinstance(data_obj, dict): data_obj = [data_obj]
    for obj in data_obj:
        name = obj['name']
        poly = json_utils.get_value(obj, ['polygon', 'pt'], default=[])
        poly = [[float(p['x']), float(p['y'])] for p in poly]
        if len(poly) == 0: continue
        info = {"label": name, "points": poly, "group_id": 0}
        annos.append(info)
    return filename, annos, width, height


def convert_cvat2labelme(image_dir, anno_dir, vis=True):
    """
    将CVAT标注格式(LabelMe 3.0)转换labelme通用格式
    :param image_dir:
    :param anno_dire:
    :param vis:
    :return:
    """
    xml_list = file_utils.get_files_lists(anno_dir, postfix=["*.xml"])
    for xml_file in xml_list:
        image_name, annos, width, height = parser_annotations(xml_file)
        points = [an['points'] for an in annos if len(an['points']) > 0]
        labels = [an['label'] for an in annos if an['label']]
        if len(points) == 0 or len(labels) == 0:
            print("empty:{}".format(xml_file))
            continue
        image_file = os.path.join(image_dir, image_name)
        image = cv2.imread(image_file)
        h, w = image.shape[:2]
        image_id = image_name.split(".")[0]
        json_file = os.path.join(anno_dir, f"{image_id}.json")
        build_labelme.maker_labelme(json_file, points, labels, image_name, image_size=[w, h], image_bs64=None)


if __name__ == "__main__":
    image_dir = "/home/PKing/Downloads/default"
    anno_dir = image_dir
    convert_cvat2labelme(image_dir=image_dir, anno_dir=anno_dir, vis=False)
