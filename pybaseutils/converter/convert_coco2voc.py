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
from pybaseutils.dataloader import parser_coco_det
from pybaseutils.converter import build_voc
from pybaseutils import file_utils, image_utils


def convert_coco2voc(filename, out_xml_dir=None, out_image_dir=None, class_name=None, class_dict={},
                     rename="", vis=True):
    """
    将COCO格式转换为VOC格式
    :param filename:
    :param out_xml_dir: output VOC XML,Annotations
    :param out_image_dir: output VOC image if not None ,JPEGImages
    :param class_name:
    :param rename: 新名字flag
    """
    if isinstance(class_name, dict):
        class_dict = class_name
        class_name = list(class_name.keys())
    else:
        class_dict = None
        class_name = class_name
    dataset = parser_coco_det.CocoDetection(anno_file=filename,
                                            data_root=None,
                                            anno_dir=None,
                                            image_dir=None,
                                            class_name=class_name,
                                            transform=None,
                                            use_rgb=False,
                                            decode=False,
                                            shuffle=False)
    print("have num:{}".format(len(dataset)))
    print("have num:{}".format(len(dataset)))
    if class_dict: class_name = [class_dict[n] for n in class_name if n in class_dict]
    class_set = []
    for i in tqdm(range(len(dataset))):
        data = dataset.__getitem__(i)
        image, targets = data["image"], data["target"]
        image_file = data["image_file"]
        bboxes, labels = targets[:, 0:4], targets[:, 4:5]
        labels = np.asarray(labels, np.int32).reshape(-1).tolist()
        names = [dataset.class_name[i] for i in labels]
        if class_dict: names = [class_dict[n] for n in names if n in class_dict]
        image_shape = image.shape
        class_set = names + class_set
        class_set = list(set(class_set))
        if len(names) == 0 or image is None:
            print("Error:{}".format(image_file))
            continue
        # bboxes, labels = traffic_light(bboxes, labels)
        format = os.path.basename(image_file).split(".")[-1]
        image_id = os.path.basename(image_file)[:-len(format) - 1]
        if rename:
            image_id = "{}_{:0=4d}".format(rename, i)
            format = "jpg"
        newname = "{}.{}".format(image_id, format)
        xml_path = file_utils.create_dir(out_xml_dir, None, "{}.xml".format(image_id))
        objects = build_voc.create_objects(bboxes, names, keypoints=None, class_name=None)
        build_voc.write_voc_xml_objects(newname, image_shape, objects, xml_path)
        if out_image_dir:
            file_utils.copy_file_to_dir(image_file, out_image_dir)

        if vis:
            parser_coco_det.show_target_image(image, bboxes, labels, normal=False, class_name=class_name,
                                              transpose=False, use_rgb=False)
    if out_image_dir:
        file_utils.save_file_list(out_image_dir, filename=None, prefix="", postfix=file_utils.IMG_POSTFIX,
                                  only_id=False, shuffle=False, max_num=None)
    print("class_set:{}".format(class_set))


if __name__ == "__main__":
    filename = "/home/PKing/nasdata/dataset/tmp/hand-pose/HandPose-v2/train/train_anno.json"
    out_xml_dir = os.path.join(os.path.dirname(filename), "VOC/Annotations")
    out_image_dir = os.path.join(os.path.dirname(filename), "VOC/JPEGImages")
    class_name = ['hand']
    convert_coco2voc(filename, out_xml_dir, out_image_dir=out_image_dir, class_name=class_name, rename="", vis=False)
