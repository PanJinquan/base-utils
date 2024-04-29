# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2023-09-08 18:31:24
    @Brief  :
"""
import os
from pybaseutils.converter import convert_voc2voc
from pybaseutils import file_utils, image_utils


def gesture_convert_voc2voc(filename, out_root, max_num=2000, vis=False):
    out_xml_dir = os.path.join(out_root, "Annotations")
    out_img_dir = os.path.join(out_root, "JPEGImages")
    class_name = ['unique']
    class_dict = {'unique': "hand"}
    convert_voc2voc.convert_voc2voc(filename,
                                    out_xml_dir=out_xml_dir,
                                    out_img_dir=out_img_dir,
                                    class_dict=class_dict,
                                    class_name=class_name,
                                    rename="",
                                    max_num=max_num,
                                    vis=vis)


if __name__ == "__main__":
    out_root = "/home/PKing/nasdata/dataset/tmp/hand-pose/Hand-voc3"
    inp_root = "/home/PKing/nasdata/dataset/tmp/gesture/Light-HaGRID/trainval"
    sub_names = file_utils.get_sub_paths(inp_root)
    for sub in sub_names:
        filename = os.path.join(inp_root, sub, "trainval.txt")
        print(filename)
        if os.path.exists(filename):
            gesture_convert_voc2voc(filename, out_root)
        else:
            print("error:not exists:{}".format(filename))
