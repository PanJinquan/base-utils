# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : 2025-07-08 14:10:15
# @Brief  : 转换labelme标注数据为voc格式
# --------------------------------------------------------
"""
import os
from pybaseutils.converter import convert_labelme2voc
if __name__ == "__main__":
    # 将image_dir修改你labelme标注数据的根目录
    image_dir = "/home/PKing/nasdata/tmp/tmp/pen/笔尖指尖标注方法/JPEGImages"
    json_dir  = image_dir
    out_root = os.path.dirname(json_dir)
    class_dict = {}
    lm = convert_labelme2voc.Labelme2VOC(image_dir, json_dir)
    lm.build_dataset(out_root, class_dict=class_dict, vis=False, crop=False)
