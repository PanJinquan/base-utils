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


def save_file_list_labels(data_dir, class_file="", out_path=None, shuffle=False, add_sub=True):
    """
     [image_path, label]
    :param image_dir: 一类一个文件夹
    :param class_file: 如果含有class_file，会进行映射ID
    :param out_path:  保存格式[path/to/image,label]
    :param shuffle:
    :return:
    """
    sub = os.path.basename(data_dir)
    # file_list = file_utils.get_files_lists(data_dir, postfix=file_utils.IMG_POSTFIX)
    file_list = file_utils.get_files_lists(data_dir, postfix=file_utils.AUDIO_POSTFIX)
    file_list = file_utils.get_sub_list(file_list, data_dir)
    # file_list = file_utils.get_sub_list(file_list, os.path.dirname(image_dir))
    class_name = None
    if class_file and os.path.exists(class_file):
        class_name = file_utils.read_data(class_file, None)
    content_list = []
    for image_path in file_list:
        # label = os.path.basename(os.path.dirname(image_path))
        label = image_path.split(os.sep)[0]
        if class_name:
            label = class_name.index(label)
        if add_sub:
            image_path = os.path.join(sub, image_path)
        item = [image_path, label]
        content_list.append(item)
    if not out_path:
        out_path = os.path.join(os.path.dirname(data_dir), "file_id.txt")
    print("num files:{},out_path:{}".format(len(content_list), out_path))
    if shuffle:
        random.seed(100)
        random.shuffle(content_list)
    file_utils.write_data(out_path, content_list, split=",")
    return content_list


if __name__ == '__main__':
    data_dir = "/home/PKing/nasdata/tmp/tmp/challenge/旋转机械故障诊断挑战赛/旋转机械故障诊断挑战赛公开数据/test"
    class_file = "/home/PKing/nasdata/tmp/tmp/challenge/旋转机械故障诊断挑战赛/旋转机械故障诊断挑战赛公开数据/class_name.txt"
    save_file_list_labels(data_dir, class_file=class_file)
