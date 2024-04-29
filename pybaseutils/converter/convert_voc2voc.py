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
from pybaseutils.converter import build_voc
from pybaseutils import file_utils, image_utils


def convert_voc2voc(filename,
                    out_xml_dir=None,
                    out_img_dir=None,
                    class_dict=None,
                    class_name=None,
                    rename="",
                    max_num=-1,
                    vis=True):
    """
    将VOC格式转换为VOC格式，以便重新label重新映射
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
    class_set = []
    new_class_name = list(set(class_dict.values()))
    nums = min(len(dataset), max_num) if max_num > 0 else len(dataset)
    for i in tqdm(range(nums)):
        data = dataset.__getitem__(i)
        image, targets, image_id = data["image"], data["target"], data["image_id"]
        bboxes, labels = targets[:, 0:4], targets[:, 4:5]
        labels = np.asarray(labels, np.int32).reshape(-1).tolist()
        labels = [dataset.class_name[i] for i in labels]
        image_shape = image.shape
        image_file = data["image_file"]
        class_set = labels + class_set
        class_set = list(set(class_set))
        if len(labels) == 0 or image is None:
            print("Error:{}".format(image_file))
            continue
        if class_dict:
            labels = [class_dict[n] for n in labels]
        # bboxes, labels = traffic_light(bboxes, labels)
        format = os.path.basename(image_file).split(".")[-1]
        image_id = os.path.basename(image_file)[:-len(format) - 1]
        if rename:
            image_id = "{}_{:0=4d}".format(rename, i)
            format = "jpg"
        newname = "{}.{}".format(image_id, format)
        xml_path = file_utils.create_dir(out_xml_dir, None, "{}.xml".format(image_id))
        objects = build_voc.create_objects(bboxes, labels, keypoints=None, class_name=None)
        build_voc.write_voc_xml_objects(newname, image_shape, objects, xml_path)
        if out_img_dir:
            dst_file = file_utils.create_dir(out_img_dir, None, newname)
            # file_utils.copy_file(image_file, dst_file)
            cv2.imwrite(dst_file, image)

        if vis:
            labels = [new_class_name.index(l) for l in labels]
            parser_voc.show_target_image(image, bboxes, labels, normal=False, class_name=new_class_name,
                                         transpose=False, use_rgb=False)
    if out_img_dir:
        file_utils.save_file_list(out_img_dir, filename=None, prefix="", postfix=file_utils.IMG_POSTFIX,
                                  only_id=False, shuffle=False, max_num=None)
    print("class_set:{}".format(class_set))


if __name__ == "__main__":
    filename = "/home/PKing/nasdata/dataset/tmp/gesture/Light-HaGRID/trainval/dislike/trainval.txt"
    out_root = "/home/PKing/nasdata/dataset/tmp/hand-pose/Hand-voc3"
    out_xml_dir = os.path.join(out_root, "Annotations")
    out_img_dir = os.path.join(out_root, "JPEGImages")
    class_name = ['unique']
    class_dict = {'unique': "hand"}
    convert_voc2voc(filename, out_xml_dir, out_img_dir=out_img_dir, class_name=class_name, class_dict=class_dict,
                    rename="",
                    vis=False)
