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
    words_list = file_utils.read_data(file, split=None)
    words_dict = {}
    count = 0
    for ws in words_list:
        # ws = font_utils.get_string_chinese(ws)
        for w in ws:
            assert w not in words_dict, Exception(f"Error:该组:[{ws}]的字:[{w}],在其他组中存在了,请合并")
            words_dict[w] = count
            count += 1
    words = list(words_dict.keys())
    print(words_dict)
    print("have:{}".format(len(words_dict)))
    filename = file_utils.create_dir(os.path.dirname(file), None, "word.txt")
    file_utils.write_list_data(filename, words)
    print("num files:{},out_path:{}".format(len(words), filename))
    return words_dict


def filter_words(classes, target):
    classes = [c for c in classes if c in target]
    return classes


if __name__ == "__main__":
    file = "/home/dm/cv/panjinquan/dataset-dmai/handwriting/word-class/trainval/similar/形近字表v2.txt"
    classes = read_word_file(file)
