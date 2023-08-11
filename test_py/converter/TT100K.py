# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2023-04-10 11:43:24
    @Brief  : https://cg.cs.tsinghua.edu.cn/traffic-sign/tutorial.html
"""
import os
import json
import cv2
import tt100k_utils
from tqdm import tqdm
from pybaseutils import image_utils, file_utils
from pybaseutils.converter import build_voc


def tt100k(data_file, anno_file, out_voc, vis=False):
    print("out_voc   :{}".format(out_voc))
    datadir = os.path.dirname(os.path.dirname(data_file))
    phase = os.path.basename(os.path.dirname(data_file))
    image_ids = open(data_file).read().splitlines()
    annos = json.loads(open(anno_file).read())
    if out_voc:
        out_image_dir = file_utils.create_dir(out_voc, phase, "JPEGImages")
        out_xml_dir = file_utils.create_dir(out_voc, phase, "Annotations")
    class_set = []
    for image_id in tqdm(image_ids):
        img = annos["imgs"][image_id]
        image_file = os.path.join(datadir, img['path'])
        image_name = os.path.basename(image_file)
        image_id = image_name.split(".")[0]
        image = cv2.imread(image_file)
        image_shape = image.shape
        bboxes, labels, vis_image, mask = tt100k_utils.draw_all(annos, datadir, image_id, image)
        if len(labels) == 0: continue
        class_set = labels + class_set
        class_set = list(set(class_set))
        if out_voc:
            xml_path = file_utils.create_dir(out_xml_dir, None, "{}.xml".format(image_id))
            jpg_path = file_utils.create_dir(out_image_dir, None, image_name)
            objects = build_voc.create_objects(bboxes, labels)
            build_voc.write_voc_xml_objects(image_name, image_shape, objects, xml_path)
            # file_utils.copy_file(image_file, jpg_path)
            cv2.imwrite(jpg_path, image)
        if vis:
            image = image_utils.draw_image_bboxes_text(image, bboxes, labels,
                                                       color=(255, 0, 0), thickness=2, fontScale=1.0)
            image_utils.cv_show_image("image", image, use_rgb=False)
    out_file = os.path.join(out_voc, phase, "class_names.txt")
    class_set.sort()
    file_utils.write_list_data(out_file, class_set)
    print("class_set:{}".format(class_set))


if __name__ == '__main__':
    out_voc = "/home/dm/nasdata/dataset/tmp/traffic-sign/TT100K/VOC/"
    data_file = "/home/dm/nasdata/dataset/tmp/traffic-sign/TT100K/data/train/ids.txt"
    anno_file = "/home/dm/nasdata/dataset/tmp/traffic-sign/TT100K/data/annotations.json"
    tt100k(data_file, anno_file, None, vis=True)
