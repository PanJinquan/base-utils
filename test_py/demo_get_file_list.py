# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: python-learning-notes
# @Author : Pan
# @E-mail : 390737991@qq.com
# @Date   : 2020-06-18 21:53:04
# --------------------------------------------------------
"""
import os
import random
from tqdm import tqdm
from pybaseutils import file_utils, image_utils


def save_file_list_labels(image_dir, class_file="", out_path=None, shuffle=False, add_sub=True):
    """
     [image_path, label]
    :param image_dir: 一类一个文件夹
    :param class_file: 如果含有class_file，会进行映射ID
    :param out_path:  保存格式[path/to/image,label]
    :param shuffle:
    :return:
    """
    sub = os.path.basename(image_dir)
    image_list = file_utils.get_files_lists(image_dir, postfix=["*.jpg", "*.png"])
    # image_list = file_utils.get_files_lists(image_dir, postfix=["*.json"])
    image_list = file_utils.get_sub_list(image_list, image_dir)
    # image_list = file_utils.get_sub_list(image_list, os.path.dirname(image_dir))
    class_name = None
    if class_file and os.path.exists(class_file):
        class_name = file_utils.read_data(class_file, None)
    content_list = []
    for image_path in image_list:
        # label = os.path.basename(os.path.dirname(image_path))
        label = image_path.split(os.sep)[0]
        if class_name:
            label = class_name.index(label)
        if add_sub:
            image_path = os.path.join(sub, image_path)
        item = [image_path, label]
        content_list.append(item)
    if not out_path:
        out_path = os.path.join(os.path.dirname(image_dir), "file_id.txt")
    print("num files:{},out_path:{}".format(len(content_list), out_path))
    if shuffle:
        random.seed(100)
        random.shuffle(content_list)
    file_utils.write_data(out_path, content_list, split=",")
    return content_list


def save_file_list(file_dir, prefix="", postfix=file_utils.IMG_POSTFIX, only_id=True, shuffle=False, max_num=None):
    """保存文件列表"""
    filename = os.path.join(os.path.dirname(file_dir), "file_list.txt")
    file_list = file_utils.get_files_list(file_dir, prefix=prefix, postfix=postfix, basename=False)
    file_list = file_utils.get_sub_list(file_list, dirname=file_dir)
    if only_id:
        file_list = [str(f).split(".")[0] for f in file_list]
    if shuffle:
        random.seed(100)
        random.shuffle(file_list)
    if max_num:
        max_num = min(max_num, len(file_list))
        file_list = file_list[0:max_num]
    file_utils.write_list_data(filename, file_list)
    print("num files:{},out_path:{}".format(len(file_list), filename))


def get_voc_file_list(voc_root,
                      prefix="",
                      postfix=file_utils.IMG_POSTFIX,
                      only_id=True,
                      check=True,
                      shuffle=False,
                      max_num=None):
    """
    获得VOC数据集的文件列表，并剔除无效的文件
    :param voc_root:
    :param prefix:
    :param postfix:
    :param only_id:
    :param check: 检测xml和图片是否存在
    :param shuffle:
    :param max_num:
    :return:
    """
    image_dir = os.path.join(voc_root, "JPEGImages")
    annos_dir = os.path.join(voc_root, "Annotations")
    filename = os.path.join(os.path.dirname(image_dir), "file_list.txt")
    file_list = file_utils.get_files_list(image_dir, prefix=prefix, postfix=postfix, basename=False)
    file_list = file_utils.get_sub_list(file_list, dirname=image_dir)
    if check:
        xmls_list = file_utils.get_files_list(annos_dir, postfix=["*.xml"], basename=True)
        print("xml file:{},image file:{}".format(len(xmls_list), len(file_list)))
        xmls_ids = [str(f).split(".")[0] for f in xmls_list]
        file_list = [f for f in file_list if str(f).split(".")[0] in xmls_ids]
    if only_id:
        file_list = [str(f).split(".")[0] for f in file_list]
    if shuffle:
        random.seed(100)
        random.shuffle(file_list)
    if max_num:
        max_num = min(max_num, len(file_list))
        file_list = file_list[0:max_num]
    file_utils.write_list_data(filename, file_list)
    print("num files:{},out_path:{}".format(len(file_list), filename))


if __name__ == "__main__":
    # file_dir = "/home/PKing/cv/panjinquan/dataset/face_person/AIJE/outdoor/JPEGImages"
    # save_file_list(file_dir, postfix=file_utils.IMG_POSTFIX + ["*.JPG"], only_id=False, shuffle=False, max_num=None)
    voc_root = "/home/PKing/Downloads/MTFL/VOC"
    get_voc_file_list(voc_root, postfix=file_utils.IMG_POSTFIX + ["*.JPG"], only_id=False, shuffle=False, max_num=None)
