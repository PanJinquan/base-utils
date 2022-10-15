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
from pybaseutils.dataloader import parser_voc
from pybaseutils import file_utils, image_utils, coords_utils


def create_texts(bboxes: np.asarray, labels: list, width, height, class_name={}):
    """
    :param bboxes: [[xmin,ymin,xmax,ymax]]
    :param labels:[name1,name2]
    :param keypoint:
    :return:
    """
    contents = []
    centers = coords_utils.xyxy2cxcywh(bboxes, width=width, height=height, normalized=True)
    for i in range(len(labels)):
        label = class_name[labels[i]] if class_name else labels[i]
        center = centers[i].tolist()
        item = [label] + center
        contents.append(item)
    return contents


def convert_voc2text(filename, out_text_dir, out_image_dir=None, class_name=None, rename="", vis=True):
    """将VOC的(xmin,ymin,xmax,ymax)转换为YOLO格式数据(class,cx,cy,w,h)/(1,width,height,width,height)
    :param filename:
    :param out_text_dir: output VOC XML
    :param out_image_dir: output VOC image if not None
    :param class_name: 如{0: "face", 1: "person"} label-map  if not None
    :param flag: 新名字flag
    """
    dataset = parser_voc.VOCDatasets(filename=[filename],
                                     data_root=None,
                                     anno_dir=None,
                                     image_dir=None,
                                     class_name=class_name,
                                     transform=None,
                                     use_rgb=False,
                                     check=False,
                                     shuffle=False)
    print("have num:{}".format(len(dataset)))
    for i in tqdm(range(len(dataset))):
        data = dataset.__getitem__(i)
        image, targets, image_id = data["image"], data["target"], data["image_id"]
        bboxes, labels = targets[:, 0:4], targets[:, 4:5]
        h, w = image.shape[:2]
        image_file = data["image_file"]
        if len(labels) == 0 or image is None:
            print("Error:{}".format(image_file))
            continue
        format = os.path.basename(image_file).split(".")[-1]
        image_id = os.path.basename(image_file)[:-len(format)-1]
        if rename:
            image_id = "{}_{:0=4d}".format(rename, i)
            format = "jpg"
        newname = "{}.{}".format(image_id, format)
        text_path = file_utils.create_dir(out_text_dir, None, "{}.txt".format(image_id))
        labels = np.asarray(labels).astype(np.int).reshape(-1)
        contents = create_texts(bboxes, labels, width=w, height=h)
        file_utils.write_data(text_path, contents)
        if out_image_dir:
            dst_file = file_utils.create_dir(out_image_dir, None, newname)
            file_utils.copy_file(image_file, dst_file)

        if vis:
            # if class_name: labels = [class_name[l] for l in labels]
            parser_voc.show_target_image(image, bboxes, labels, normal=False,
                                         transpose=False, use_rgb=False)


if __name__ == "__main__":
    # filename = "/home/dm/nasdata/dataset/csdn/helmet/helmet-asian/total.txt"
    filename = "/home/dm/nasdata/dataset/csdn/helmet/Hard Hat Workers.v2-raw.voc/VOC/trainval.txt"
    out_text_dir = os.path.join(os.path.dirname(filename), "labels")
    # class_name = {'person': 0, 'hat': 1}
    # class_name = {'person': 0, 'head': 1, "helmet": 2}
    class_name = {'head': 0, "helmet": 1}
    convert_voc2text(filename, out_text_dir, out_image_dir=None, class_name=class_name,rename="", vis=False)
