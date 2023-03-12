# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  :
"""
import os
import cv2
import time
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
from pybaseutils import file_utils, image_utils, base64_utils, time_utils, font_utils, json_utils


def read_word_file(file):
    words = file_utils.read_data(file, split=None)
    words_ = []
    for ws in words:
        ws = font_utils.get_string_chinese(ws)
        ws = [w for w in ws]
        words_ += ws
    words_ = list(set(words_))
    return words_


def filter_words(classes, target):
    classes = [c for c in classes if c in target]
    return classes


if __name__ == "__main__":
    file = "/home/dm/nasdata/dataset-dmai/handwriting/word-class/trainval/class_name3594.txt"
    file1 = "/home/dm/nasdata/dataset-dmai/handwriting/word-class/word-similar/【DMICE0084】书法评测研发项目-字库v4形近字表梳理_第二批_1135字.txt"
    file2 = "/home/dm/nasdata/dataset-dmai/handwriting/word-class/word-similar/【DMICE0084】书法评测研发项目-字库v4形近字表梳理_第一批_2459字.txt"
    target = read_word_file(file)
    class1 = read_word_file(file1)
    class2 = read_word_file(file2)
    classes = list(set(class1 + class2))
    classes = filter_words(classes, target)
    print(len(classes))
    print(classes)
