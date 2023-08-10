# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: python-learning-notes
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-02-21 10:15:42
# --------------------------------------------------------
"""

import os
import argparse
import glob
import copy as copy
from modules.dataset_tool.coco_tools.convert_voc2coco import save_json, read_json, COCOTools


class ConcatCocoDataset(object):
    """
    拼接多个COCO数据集文件夹
    PS :保证coco的image的id和file_name唯一的
    """

    def __init__(self):
        # init COCO Dataset struct
        self.coco = {
            "images": [
            ],
            "annotations": [
            ],
            "categories": [
            ]
        }

    def add_categories(self, categories):
        """
        Add categories to COCO["categories"]
        :param categories:
        :return:
        """
        categories_id = COCOTools.get_categories_id(self.coco["categories"])
        cat_name = list(categories_id.keys())
        cat_id = list(categories_id.values())
        max_id = 0
        if cat_id:
            max_id = max(cat_id)
        for item in categories:
            name = item["name"]
            id = item["id"]
            if name in cat_name:
                continue
            max_id += 1
            item["id"] = max_id  # BUG:will assign to categories
            self.coco['categories'].append(item)

    def add_images(self, images, dirname=None):
        """
        Add images to COCO["images"]
        :param images:
        :param dirname:由于多个数据集合并的图片数据保存路径不同,
                    这里提供dirname(父级目录)用于区分不同数据
        :return:
        """
        images_id = COCOTools.get_images_id(self.coco["images"])
        add_images_id = COCOTools.get_images_id(images)
        for id in add_images_id:
            assert id not in images_id, Exception("have same image_id:{}".format(id))

        for item in images:
            if dirname:
                item["dirname"] = dirname
                # item["file_name"] = os.path.join(dirname, item["file_name"])
            self.coco['images'].append(item)

    def add_annotations(self, annotations, categories):
        """
        Add annotations to COCO["annotations"]
        :param annotations:
        :param categories:
        :return:
        """
        annotations_id = COCOTools.get_annotations_id(self.coco["annotations"])
        categories_id = COCOTools.get_categories_id(self.coco["categories"])
        # cat_name = categories_id.keys()
        # cat_id = categories_id.values()
        max_id = 0
        if annotations_id:
            max_id = max(annotations_id)
        add_categories_id = COCOTools.get_categories_id(categories)
        add_id_categories = {v: k for k, v in add_categories_id.items()}

        for item in annotations:
            category_id = item["category_id"]
            name = add_id_categories[category_id]
            item["category_id"] = categories_id[name]
            max_id += 1
            item["id"] = max_id
            self.coco['annotations'].append(item)
            # annotations_id__ = self.get_annotations_id(self.coco["annotations"])
            # self.check_uniqueness(annotations_id, title="annotations_id")

    def merge_coco_dataset(self, file_dict, save_json_path, check=True):
        """
        :param file_dict: if dict:{ dirname1:coco_file1.json,
                                    dirname2:coco_file2.json,
                                  }
                          if list:[coco_file1.json,coco_file2.json,...]
        :param save_json_path:
        :param check whether to check,default is True
        :return:
        """
        if isinstance(file_dict, dict):
            for dirname, file in file_dict.items():
                coco = read_json(file)
                self.add_categories(copy.deepcopy(coco["categories"]))
                self.add_images(copy.deepcopy(coco["images"]), dirname)
                self.add_annotations(copy.deepcopy(coco["annotations"]),
                                     copy.deepcopy(coco["categories"]))
        elif isinstance(file_dict, list):
            for file in file_dict:
                coco = read_json(file)
                self.add_categories(copy.deepcopy(coco["categories"]))
                self.add_images(copy.deepcopy(coco["images"]), dirname=None)
                self.add_annotations(copy.deepcopy(coco["annotations"]),
                                     copy.deepcopy(coco["categories"]))
        if check:
            COCOTools.check_coco(self.coco)
        save_json(self.coco, save_json_path)


if __name__ == '__main__':
    # file_dict = {
    #     "lexue_val": "/media/dm/dm/project/dataset/coco/annotations/lexue/lexue_val.json",
    #     "val2017": "/media/dm/dm/project/dataset/coco/annotations/person_keypoints_val2017.json"}
    # save_json_path = "/media/dm/dm/project/dataset/coco/annotations/lexue/person_keypoints_val2017_lexue_val.json"


    file_dict = {
        "lexue_train": "/media/dm/dm/project/dataset/coco/annotations/lexue/lexue_train.json",
        "lexue_val": "/media/dm/dm/project/dataset/coco/annotations/lexue/lexue_val.json"}
    # file_dict = list(file_dict.values())
    save_json_path = "/media/dm/dm/project/dataset/coco/annotations/lexue/lexue_train_val.json"
    #
    ccd = ConcatCocoDataset()
    ccd.merge_coco_dataset(file_dict, save_json_path)
