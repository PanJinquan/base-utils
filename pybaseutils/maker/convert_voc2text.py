# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-10-12 16:21:55
    @Brief  :
"""
import os
import numpy as np
from tqdm import tqdm
from pybaseutils.dataloader import parser_textdata
from pybaseutils.maker import maker_voc
from pybaseutils import file_utils, image_utils


def convert_voc2text(filename, out_xml_dir, out_image_dir=None, class_name=None, flag="", vis=True):
    """将YOLO格式数据[class cx cy w h]转换为VOC[xmin,ymin,xmax,ymax]
    :param filename:
    :param out_xml_dir: output VOC XML
    :param out_image_dir: output VOC image if not None
    :param class_name: 如{0: "face", 1: "person"} label-map  if not None
    :param flag: 新名字flag
    """
    dataset = parser_textdata.TextDataset(filename=filename,
                                          data_root=None,
                                          anno_dir=None,
                                          image_dir=None,
                                          class_name=None,
                                          check=False,
                                          phase="val",
                                          shuffle=False)
    print("have num:{}".format(len(dataset)))
    for i in tqdm(range(len(dataset))):
        data = dataset.__getitem__(i)
        image, points, bboxes, labels = data["image"], data["point"], data["box"], data["label"]
        h, w = image.shape[:2]
        image_shape = image.shape
        image_file = data["image_file"]
        image_id = os.path.basename(image_file).split(".")[0]
        image_id = "{}_{}".format(image_id, flag) if flag else image_id
        if len(labels) == 0 or image is None:
            print("Error:{}".format(image_file))
            continue
        format = os.path.basename(image_file).split(".")[-1]
        newname = "{}.{}".format(image_id, format)
        xml_path = file_utils.create_dir(out_xml_dir, None, "{}.xml".format(image_id))
        labels = np.asarray(labels).astype(np.int).reshape(-1)
        objects = maker_voc.create_objects(bboxes, labels, keypoints=None, class_name=class_name)
        maker_voc.write_voc_xml_objects(newname, image_shape, objects, xml_path)
        if out_image_dir:
            dst_file = file_utils.create_dir(out_image_dir, None, newname)
            file_utils.copy_file(image_file, dst_file)

        if vis:
            if class_name: labels = [class_name[l] for l in labels]
            parser_textdata.show_target_image(image, bboxes, labels, points)


if __name__ == "__main__":
    filename = "/home/dm/nasdata/dataset/csdn/helmet/helmet-dataset-v2/train.txt"
    out_xml_dir = "/home/dm/nasdata/dataset/csdn/helmet/helmet-dataset-v2/VOC/Annotations"
    out_image_dir = "/home/dm/nasdata/dataset/csdn/helmet/helmet-dataset-v2/VOC/JPEGImages"
    flag = "v1"
    class_name = {0: "hat", 1: "person"}
    convert_voc2text(filename, out_xml_dir, out_image_dir=out_image_dir, class_name=class_name, flag=flag, vis=False)
