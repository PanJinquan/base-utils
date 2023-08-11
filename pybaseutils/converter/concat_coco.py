# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project:
# @Author : panjq
# @Date   : 2020-02-12 18:28:16
# @url    :
# --------------------------------------------------------
"""

import os
import argparse
import glob
import copy as copy
from pybaseutils.converter.build_coco import COCOTools
from pybaseutils import file_utils


class ConcatCoco(object):
    """
    拼接多个COCO数据集文件夹
    PS :保证coco的image的id和file_name唯一的
    """

    def __init__(self):
        self.coco = {"images": [], "annotations": [], "categories": [], "type": "instances"}

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
        rec_images_id = COCOTools.get_images_id(self.coco["images"])  # coco中已经存在的image_id
        add_images_id = COCOTools.get_images_id(images)  # 需要新增的image_id
        for id in add_images_id:
            assert id not in rec_images_id, Exception("have same image_id:{}".format(id))

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

    def cancat_coco_dataset(self, file_dict, check=True):
        """
        :param file_dict: if dict:{ dirname1:coco_file1.json,
                                    dirname2:coco_file2.json,
                                  }
                          if list:[coco_file1.json,coco_file2.json,...]
        :param check whether to check,default is True
        :return:
        """
        if isinstance(file_dict, dict):
            for dirname, file in file_dict.items():
                coco = file_utils.read_json_data(file)
                self.add_categories(copy.deepcopy(coco["categories"]))
                self.add_images(copy.deepcopy(coco["images"]), dirname)
                self.add_annotations(copy.deepcopy(coco["annotations"]),
                                     copy.deepcopy(coco["categories"]))
        elif isinstance(file_dict, list):
            for file in file_dict:
                coco = file_utils.read_json_data(file)
                self.add_categories(copy.deepcopy(coco["categories"]))
                self.add_images(copy.deepcopy(coco["images"]), dirname=None)
                self.add_annotations(copy.deepcopy(coco["annotations"]),
                                     copy.deepcopy(coco["categories"]))
        if check:
            COCOTools.check_coco(self.coco)

    def save_coco(self, json_file):
        file_utils.create_file_path(json_file)
        file_utils.write_json_path(json_file, self.coco)
        print("save file:{}".format(json_file))


if __name__ == '__main__':
    file_dict = {
        "person1": "/media/PKing/新加卷1/SDK/base-utils/data/person1.json",
        "person2": "/media/PKing/新加卷1/SDK/base-utils/data/person2.json"}

    save_coco_file = "/media/PKing/新加卷1/SDK/base-utils/data/merge_person.json"
    build = ConcatCoco()
    build.cancat_coco_dataset(file_dict)
    build.save_coco(save_coco_file)

