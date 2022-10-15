# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-10-12 16:21:55
    @Brief  :
"""
import os
import numpy as np
import cv2
from tqdm import tqdm
from pybaseutils.dataloader import parser_textdata
from pybaseutils.maker import maker_voc
from pybaseutils import file_utils, image_utils


def convert_text2voc(filename, out_xml_dir, out_image_dir=None, class_name=None, rename="", vis=True):
    """将YOLO格式数据(class,cx,cy,w,h)/(1,width,height,width,height)转换为VOC(xmin,ymin,xmax,ymax)
    :param filename:
    :param out_xml_dir: output VOC XML,Annotations
    :param out_image_dir: output VOC image if not None ,JPEGImages
    :param class_name: 如{0: "face", 1: "person"} label-map  if not None
    :param rename: 新名字flag
    """
    dataset = parser_textdata.TextDataset(filename=filename,
                                          data_root=None,
                                          anno_dir=None,
                                          image_dir=None,
                                          class_name=None,
                                          use_rgb=False,
                                          check=False,
                                          phase="val",
                                          shuffle=False)
    print("have num:{}".format(len(dataset)))
    for i in tqdm(range(len(dataset))):
        data = dataset.__getitem__(i)
        image, bboxes, labels = data["image"], data["box"], data["label"]
        image_shape = image.shape
        image_file = data["image_file"]
        if len(labels) == 0 or image is None:
            print("Error:{}".format(image_file))
            continue
        format = os.path.basename(image_file).split(".")[-1]
        image_id = os.path.basename(image_file)[:-len(format) - 1]
        if rename:
            image_id = "{}_{:0=4d}".format(rename, i)
            format = "jpg"
        newname = "{}.{}".format(image_id, format)
        xml_path = file_utils.create_dir(out_xml_dir, None, "{}.xml".format(image_id))
        labels = np.asarray(labels).astype(np.int).reshape(-1)
        objects = maker_voc.create_objects(bboxes, labels, keypoints=None, class_name=class_name)
        maker_voc.write_voc_xml_objects(newname, image_shape, objects, xml_path)
        if out_image_dir:
            dst_file = file_utils.create_dir(out_image_dir, None, newname)
            # file_utils.copy_file(image_file, dst_file)
            cv2.imwrite(dst_file, image)

        if vis:
            parser_textdata.show_target_image(image, bboxes, labels, class_name=class_name, use_rgb=False)


if __name__ == "__main__":
    filename = "/home/dm/nasdata/dataset/csdn/helmet/Helmet-Asian/total.txt"
    out_xml_dir = os.path.join(os.path.dirname(filename), "VOC/Annotations")
    # out_image_dir = os.path.join(os.path.dirname(filename),"VOC/JPEGImages")
    # out_image_dir = None
    class_name = {0: 'head', 1: "helmet"}
    convert_text2voc(filename, out_xml_dir, out_image_dir=None, class_name=class_name, rename="", vis=False)
