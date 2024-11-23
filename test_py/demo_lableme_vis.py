# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2024-05-23 11:24:37
    @Brief  :
"""
import os
from pybaseutils import file_utils, image_utils
from pybaseutils.dataloader import parser_labelme
from pybaseutils.converter import build_labelme

if __name__ == "__main__":

    anno_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-indoor-det/dataset-v1/json"
    anno_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-indoor-det/dataset-v7/json"
    anno_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/使用钳形电流表测量低压线路电流/dataset-v1/json"
    # anno_dir = [anno_dir, anno_dir]
    names = ['A相电线', 'B相电线', 'C相电线', 'N相电线']

    # anno_dir = "/home/PKing/nasdata/tmp/tmp/WaterMeter/水表数据集/Water-Meter-Det3/train/json"
    # names = ["BG", "unique"]
    anno_dir = "/home/PKing/nasdata/tmp/tmp/WaterMeter/水表数据集/Water-Meter-Det1/val/json"
    # anno_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-outdoor-det-fix/dataset-v01/json"
    # anno_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-outdoor-det-fix/dataset-v23/json"
    anno_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-indoor-det/dataset-test/json"
    anno_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/dataset-v25/json"
    anno_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-negetive/dataset-v02/json"
    anno_dir = "/home/PKing/Downloads/aije-negetive/dataset-v01/sample"
    names = None
    # names = ['身穿工作服,未穿工作服', '手,手穿绝缘手套,手穿棉纱手套,手穿其他手套']
    dataset = parser_labelme.LabelMeDatasets(filename=None,
                                             data_root=None,
                                             anno_dir=anno_dir,
                                             image_dir=None,
                                             class_name=names,
                                             check=True,
                                             phase="val",
                                             shuffle=True)
    class_name = dataset.class_name
    class_file = os.path.join(os.path.dirname(anno_dir), "class_name.txt")
    file_utils.write_list_data(class_file, class_name)
    print("have num:{}".format(len(dataset)))
    for i in range(len(dataset)):
        print(i)  # i=20
        data = dataset.__getitem__(i)
        image, points, bboxes, labels = data["image"], data["points"], data["boxes"], data["labels"]
        h, w = image.shape[:2]
        if class_name: labels = [class_name[l] for l in labels]
        image_file = data["image_file"]
        anno_file = os.path.join("masker", "{}.json".format(os.path.basename(image_file).split(".")[0]))
        print(image_file)
        result = parser_labelme.show_target_image(image, bboxes, labels, points, thickness=2)
        # image_utils.save_image("./"+os.path.basename(image_file), result)
