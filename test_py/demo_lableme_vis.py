# -*- coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2024-05-23 11:24:37
    @Brief  :
"""
import os
from tqdm import tqdm
from pybaseutils import file_utils, image_utils
from pybaseutils.dataloader import parser_labelme
from pybaseutils.converter import build_labelme

if __name__ == "__main__":

    anno_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-indoor-det/dataset-v1/json"
    anno_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-indoor-det/dataset-v7/json"
    anno_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/使用钳形电流表测量低压线路电流/dataset-v1/json"
    # anno_dir = [anno_dir, anno_dir]
    # names = ['A相电线', 'B相电线', 'C相电线', 'N相电线']

    # anno_dir = "/home/PKing/nasdata/tmp/tmp/WaterMeter/水表数据集/Water-Meter-Det3/train/json"
    # names = ["BG", "unique"]
    anno_dir = "/home/PKing/nasdata/tmp/tmp/WaterMeter/水表数据集/Water-Meter-Det1/val/json"
    # anno_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-outdoor-det-fix/dataset-v01/json"
    # anno_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-outdoor-det-fix/dataset-v23/json"
    anno_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-indoor-det/dataset-test/json"
    anno_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/dataset-v25/json"
    anno_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-negetive/dataset-v02/json"
    # anno_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-date/date-det/dataset-v01/images"
    anno_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-action-cvlm-v2/train-v2/01-核相操作/dataset-test/images"
    anno_dir = [
        # TODO 训练集
        '/home/PKing/nasdata/dataset-dmai/AILT/ailt-det/dataset-v01/images',
        '/home/PKing/nasdata/dataset-dmai/AILT/ailt-det/dataset-v01-val/images',
        '/home/PKing/nasdata/dataset-dmai/AILT/ailt-det/dataset-v02/images',
        '/home/PKing/nasdata/dataset-dmai/AILT/ailt-det/dataset-v02-val/images',
        '/home/PKing/nasdata/dataset-dmai/AILT/ailt-det/dataset-v03/images',
    ]
    names = None
    dataset = parser_labelme.LabelMeDatasets(filename=None,
                                             data_root=None,
                                             anno_dir=anno_dir,
                                             image_dir=None,
                                             class_name=names,
                                             check=True ,
                                             phase="val",
                                             shuffle=False)
    class_name = dataset.class_name
    class_file = os.path.join(os.path.dirname(anno_dir[0]), "class_name.txt")
    file_utils.write_list_data(class_file, class_name)
    print("have num:{}".format(len(dataset)))
    for i in tqdm(range(len(dataset))):
        print(i)  # i=20
        data = dataset.__getitem__(i)
        image, points, bboxes, names = data["image"], data["points"], data["boxes"], data["names"]
        h, w = image.shape[:2]
        # if class_name: labels = [class_name[l] for l in labels]
        image_file = data["image_file"]
        anno_file = os.path.join("masker", "{}.json".format(os.path.basename(image_file).split(".")[0]))
        print(image_file, names)
        result = parser_labelme.show_target_image(image, bboxes, names, points, thickness=2)
        # image_utils.save_image("./"+os.path.basename(image_file), result)
