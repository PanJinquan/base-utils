# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  :
"""

import numpy as np
import xmltodict
from pybaseutils import image_utils
import cv2


def read_xml2json(xml_file):
    """
    import xmltodict
    :param xml_file:
    :return:
    """
    with open(xml_file, encoding='utf-8') as fd:  # 将XML文件装载到dict里面
        content = xmltodict.parse(fd.read())
    return content

def parse_labelmexml(anno_file):
    data = read_xml2json(anno_file)
    return None


if __name__ == '__main__':
    anno_file = "/home/PKing/Downloads/labelme/default/hard_hat_workers7.xml"
    parse_labelmexml(anno_file)
