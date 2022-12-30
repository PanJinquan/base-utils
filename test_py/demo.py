# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-21 17:34:38
    @Brief  :
"""
import time
import xmltodict
from pybaseutils import time_utils


def read_xml2json(xml_file):
    """
    import xmltodict
    :param xml_file:
    :return:
    """
    with open(xml_file, encoding='utf-8') as fd:  # 将XML文件装载到dict里面
        content = xmltodict.parse(fd.read())
    return content


if __name__ == "__main__":
    data = [[1, 2], [3, 4, 5]]
    out = []
    for d in data:
        out += d
    print(list(out))
