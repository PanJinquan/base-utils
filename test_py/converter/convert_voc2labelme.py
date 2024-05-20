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
from pybaseutils import file_utils, image_utils


def convert_voc2labelme(filename,
                        out_root=None,
                        class_name=None,
                        max_num=-1,
                        vis=True):
    """
    将VOC格式转换为labelme格式，以便重新label重新映射
    :param filename:
    :param out_xml_dir: output VOC XML,Annotations
    :param out_img_dir: output VOC image if not None ,JPEGImages
    :param class_name: 如{0: "face", 1: "person"} label-map  if not None
    :param rename: 新名字flag
    """
    dataset = parser_voc.VOCDataset(filename=filename,
                                    data_root=None,
                                    anno_dir=None,
                                    image_dir=None,
                                    class_name=class_name,
                                    transform=None,
                                    use_rgb=False,
                                    check=False,
                                    shuffle=False)
    print("have num:{}".format(len(dataset)))
    print("have num:{}".format(len(dataset)))
    nums = min(len(dataset), max_num) if max_num > 0 else len(dataset)
    for i in tqdm(range(nums)):
        data = dataset.__getitem__(i)
        image, bboxes, labels = data["image"], data["boxes"], data["labels"]
        labels = np.asarray(labels, np.int32).reshape(-1).tolist()
        labels = [dataset.class_name[i] for i in labels]
        if out_root:
            points = image_utils.boxes2polygons(bboxes)
            image_file = data["image_file"]
            image_name = os.path.basename(image_file)
            h, w = image.shape[:2]
            image_id = file_utils.get_time()

            image_name = "{}_{:0=5d}.jpg".format("image", i)
            image_id = image_name.split(".")[0]
            json_file = os.path.join(out_root, "images", f"{image_id}.json")
            img_file = os.path.join(out_root, "images", image_name)
            file_utils.copy_file(image_file, img_file)
            build_labelme.maker_labelme(json_file, points, labels, image_name, image_size=[w, h], image_bs64=None)


if __name__ == "__main__":
    filename = "/home/PKing/nasdata/tmp/tmp/WaterMeter/水表数据集/Water-Meter-Det3/train.txt"
    out_root = "/home/PKing/nasdata/tmp/tmp/WaterMeter/水表数据集/Water-Meter-Det3/labelme"
    class_name = None
    convert_voc2labelme(filename, out_root=out_root, class_name=class_name, vis=True)
