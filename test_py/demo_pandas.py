# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-10-13 18:14:53
    @Brief  :
"""
import os
import numpy as np
import cv2
from pybaseutils import pandas_utils, image_utils, file_utils
from pybaseutils.maker import maker_voc

if __name__ == "__main__":
    image_dir = "/home/dm/nasdata/dataset/csdn/helmet/Helmet_Dataset(kaggle)/helmet_dataset/JPEGImages"
    out_xml_dir = "/home/dm/nasdata/dataset/csdn/helmet/Helmet_Dataset(kaggle)/helmet_dataset/Annotations"
    file = "/home/dm/nasdata/dataset/csdn/helmet/Helmet_Dataset(kaggle)/helmet_dataset/train_labels.csv"
    df = pandas_utils.read_csv(file)
    content = pandas_utils.df2list(df)
    class_name = {'head': 0, "helmet": 1}
    vis = True
    for item in content:
        if not len(item) == 2:
            continue
        image_name, info = item
        print(image_name, info)
        info = info.lstrip().rstrip().split(" ")
        image_name = image_name.lstrip().rstrip()
        image_id, format = image_name.split(".")
        image_id = "{:0=5d}".format(int(image_id))
        image_name = "{}.{}".format(image_id, format)
        info = np.asarray(info).reshape(-1, 5)
        bboxes = np.asarray(info[:, 0:4], dtype=np.int32)
        labels_ = info[:, 4:5].reshape(-1).tolist()
        labels = []
        for name in labels_:
            name = "head" if name == "none" else "helmet"
            labels.append(name)
        image_file = os.path.join(image_dir, image_name)
        image = cv2.imread(image_file)
        image_shape = image.shape
        xml_path = file_utils.create_dir(out_xml_dir, None, "{}.xml".format(image_id))
        objects = maker_voc.create_objects(bboxes, labels, keypoints=None, class_name=None)
        maker_voc.write_voc_xml_objects(image_name, image_shape, objects, xml_path)
        if vis:
            labels = [class_name[l] for l in labels]
            image = image_utils.draw_image_bboxes_labels(image, bboxes, labels, thickness=2, fontScale=0.5)
            image_utils.cv_show_image("image", image, use_rgb=False, delay=1)
