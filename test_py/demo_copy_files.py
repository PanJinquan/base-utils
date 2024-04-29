# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-21 17:34:38
    @Brief  :
"""
import os
import time
import xmltodict
import random
from tqdm import tqdm
from pybaseutils import file_utils, image_utils


def demo_copy_move_by_sub_names_v1():
    """
    按照整个文件夹，复制或者拷贝文件
    :return:
    """
    # image_dir = "/home/dm/cv/panjinquan/dataset-dmai/handwriting/word-class/word-similar/dataset-clear/train"
    # out_dir = "/home/dm/cv/panjinquan/dataset-dmai/handwriting/word-class/word-similar/dataset-clear/test"
    # file = "/home/dm/cv/panjinquan/dataset-dmai/handwriting/word-class/word-similar/dataset-clear/形近字v1.txt"
    image_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-cls/dataset-v1/train"
    out_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-cls/dataset-v1/test"
    file = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-cls/dataset-v1/class_name.txt"
    # sub_names = ["玉", "王", "主", "玊", "壬", "玍", "生"]
    # sub_names += ["工", "土", "干", "士"]
    words = file_utils.read_data(file, split=",")
    sub_names = []
    for word in words:
        word = [w.strip() for w in word if w]  # 去除一些空格
        sub_names += word
    sub_names = list(set(sub_names))
    sub_names = sorted(sub_names)
    file_utils.copy_move_file_dir(image_dir, out_dir, sub_names=sub_names, max_nums=30*6, shuffle=True, move=True)
    out_file = os.path.join(os.path.dirname(file), "new_class_name.txt")
    file_utils.write_list_data(out_file, sub_names)


def demo_copy_move_by_sub_names_v2():
    """
    按照每个类别的个数，复制或者拷贝文件
    :return:
    """
    # image_dir = "/home/dm/cv/panjinquan/dataset-dmai/handwriting/word-class/trainval/train"
    # out_dir = "/home/dm/cv/panjinquan/dataset-dmai/handwriting/word-class/word-similar/dataset-clear/train"
    # file = "/home/dm/cv/panjinquan/dataset-dmai/handwriting/word-class/word-similar/dataset-clear/loss.txt"

    image_dir = "/home/PKing/nasdata/dataset/tmp/challenge/鸟类品种识别/鸟类品种识别挑战赛训练集/training_set"
    out_dir = "/home/PKing/nasdata/dataset/tmp/challenge/鸟类品种识别/鸟类品种识别挑战赛训练集/test"
    file = "/home/PKing/nasdata/dataset/tmp/challenge/鸟类品种识别/鸟类品种识别挑战赛训练集/class_name.txt"
    words = file_utils.read_data(file, split=",")
    sub_names = []
    for word in words:
        word = [w.strip() for w in word if w]  # 去除一些空格
        sub_names += word
    sub_names = list(set(sub_names))
    sub_names = sorted(sub_names)
    # file_utils.copy_move_dir_dir(image_dir, out_dir, sub_names=sub_names, per_nums=90, shuffle=True, move=True)
    file_utils.copy_move_dir_dir(image_dir, out_dir, sub_names=sub_names, per_nums=20, shuffle=True, move=True)
    out_file = os.path.join(os.path.dirname(file), "new_class_name.txt")
    file_utils.write_list_data(out_file, sub_names)


def demo_copy_move_by_sub_names_v3():
    image_dir = "/home/dm/nasdata/dataset/tmp/Medicine/dataset/train"
    out_dir = "/home/dm/nasdata/dataset/tmp/Medicine/dataset/test"
    # file_utils.copy_move_dir_dir(image_dir, out_dir, sub_names=sub_names, per_nums=10, shuffle=True, move=True)
    file_utils.copy_move_file_dir(image_dir, out_dir, sub_names=None, max_nums=10000, move=True, shuffle=True)


def demo_copy_move():
    image_dir = "/home/dm/nasdata/dataset-dmai/handwriting/grid-det/grid_cross_points_images/grid_cross_points_v7/儿童C组"
    out_dir = "/home/dm/nasdata/dataset-dmai/handwriting/grid-det/grid_cross_points_images/grid_cross_points_v7/images"
    file_utils.copy_move_file_dir(image_dir, out_dir, sub_names=None, max_nums=1000, move=False, shuffle=True)


def copy_files(shuffle=False):
    root = "/home/dm/nasdata/dataset/tmp/fall/Fall-detection-Dataset/train"
    out = "/home/dm/nasdata/dataset/tmp/fall/fall-v3"
    sub_list = file_utils.get_sub_paths(root)
    phase = os.path.basename(root)
    for sub in sub_list:
        image_dir = os.path.join(root, sub, "rgb")
        if not os.path.exists(image_dir):
            print("not exists:{}".format(image_dir))
            continue
        file_list = file_utils.get_images_list(image_dir)
        if shuffle:
            random.seed(100)
            random.shuffle(file_list)
            random.shuffle(file_list)
        # max_nums = len(file_list) // 3
        print("copy {}".format(image_dir))
        # if max_nums: file_list = file_list[:min(max_nums, len(file_list))]
        file_list.sort()
        interval = 10
        for count, src in enumerate(file_list):
            if count % interval == 0 and count >= 100:
                name = os.path.basename(src)
                new = "{}_{}".format(sub, name)
                dst = file_utils.create_dir(out, phase, new)
                file_utils.copy_file(src, dst)


if __name__ == "__main__":
    # demo_copy_move()
    # demo_copy_move_by_sub_names_v1()
    demo_copy_move_by_sub_names_v2()
    # demo_copy_move_by_sub_names_v2()
    # demo_copy_move_by_sub_names_v3()
    # copy_files()
    # demo_copy_move()
