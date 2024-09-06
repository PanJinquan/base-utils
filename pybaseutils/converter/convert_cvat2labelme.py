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


def convert_cvat2labelme(anno_dir, image_dir="", out_dir="", thickness=1, fontScale=1.0, vis=False):
    """
    将CVAT标注格式(LabelMe 3.0)转换labelme通用格式
    :param anno_dir: 标注文件目录，默认输出json文件与anno_dir同目录
    :param image_dir: 图片文件目录
    :param out_dir: 输出labelme *.json文件目录
    :param vis:
    :return:
    """
    if not out_dir: out_dir = os.path.join(os.path.dirname(anno_dir), "json")
    xml_list = file_utils.get_files_lists(anno_dir, postfix=["*.xml"])
    for xml_file in tqdm(xml_list):
        image_name, annos, w, h = parser_annotations(xml_file)
        image_id = os.path.basename(xml_file).split(".")[0]
        image_name = "{}.{}".format(image_id, image_name.split(".")[1])
        points = [an['points'] for an in annos if len(an['points']) > 0]
        labels = [an['label'] for an in annos if an['label']]
        if len(points) == 0 or len(labels) == 0:
            print("empty:{}".format(xml_file))
            continue
        if image_dir or vis:
            image_file = os.path.join(image_dir, image_name)
            image = cv2.imread(image_file)
            h, w = image.shape[:2]
            if vis:
                print(image_file, "labels:{}".format(labels))
                image = image_utils.draw_image_contours(image, points, texts=labels, thickness=thickness,
                                                        fontScale=fontScale)
                image_utils.cv_show_image("det", image)
        image_id = image_name.split(".")[0]
        json_file = os.path.join(out_dir, f"{image_id}.json")
        build_labelme.maker_labelme(json_file, points, labels, image_name, image_size=[w, h], image_bs64=None)


if __name__ == "__main__":
    image_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-action-cvlm-v2/aije-action-train-v02/JPEGImages"
    anno_dir = os.path.join(os.path.dirname(image_dir), "xml")
    json_dir = os.path.join(os.path.dirname(image_dir), "json")
    convert_cvat2labelme.convert_cvat2labelme(image_dir=image_dir, anno_dir=anno_dir, out_dir=json_dir,
                                              thickness=1, fontScale=2.0, vis=True)
