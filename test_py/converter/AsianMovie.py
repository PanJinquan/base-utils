# -*- coding: utf-8 -*-

import os
import sys
import random
import cv2
from tqdm import tqdm
from pybaseutils import file_utils, image_utils

class_name = {0: "neutral",
              1: "happy",
              2: "excited",
              3: "surprise",
              4: "sad",
              5: "disgust",
              6: "fear",
              7: "confused",
              8: "tired",
              9: "angry",
              10: "serious"
              }


class ParseAsian(object):

    def __init__(self, file, image_dir, shuffle=True):
        """
        0: Neutral
        1: Happy
        2: Excited
        3: Surprised
        4: Sad
        5: Disgust
        6: Fear
        7: Confused
        8: Tired
        9: Angry
        10: Serious

        ===> 0_neutral,1_happy,2_unhappy
        0      -->0
        1      -->[1,2,3]
        2      -->[4,5,6,7,8,9,10]
        id_map = {0: [0], 1: [1, 2, 3], 2: [4, 5, 6, 7, 8, 9, 10]}
        :param file:
        :param image_dir:
        :param is_detect:
        :param image_sub:
        :param shuffle:
        """
        self.image_dir = image_dir
        self.shuffle = shuffle
        self.item_list = file_utils.read_data(file)
        if shuffle:
            random.seed(200)
            random.shuffle(self.item_list)

    def parse_data(self, out_dir="", flag="20220230", max_num=None, vis=True):
        nums_sample = len(self.item_list)
        if max_num:
            nums_sample = min(max_num, nums_sample)
        dst_item_list = []
        for i in tqdm(range(nums_sample)):
            name, xmin, ymin, xmax, ymax, label = self.item_list[i]
            image_path = os.path.join(self.image_dir, name)
            if not os.path.exists(image_path):
                print("no path:{}".format(image_path))
                continue

            image = image_utils.read_image(image_path)
            name = "{}_{:0=6d}.jpg".format(flag, i)
            label = class_name[int(label)]
            item = [os.path.join(label, name), label, xmin, ymin, xmax, ymax]
            boxes = [[xmin, ymin, xmax, ymax]]
            boxes = image_utils.extend_xyxy(boxes, scale=[1.1, 1.1])
            dst_item_list.append(item)
            if out_dir:
                orig_file = file_utils.create_dir(os.path.join(out_dir, "origin"), label, name)
                crop_file = file_utils.create_dir(os.path.join(out_dir, "crops"), label, name)
                file_utils.copy_file(image_path, orig_file)
                crop = image_utils.get_bbox_crop(image, boxes[0])
                cv2.imwrite(crop_file, crop)
            if vis:
                image = image_utils.draw_image_boxes(image, boxes)
                image = image_utils.draw_texts(image, [[xmin + 5, ymin + 10]], texts=[label])
                image_utils.cv_show_image("image", image)
        filename = file_utils.create_dir(os.path.join(out_dir, "origin"), None, "label.txt")
        print(filename)
        file_utils.write_data(filename, dst_item_list, split=",")
        return dst_item_list


def main():
    file = "/home/PKing/nasdata/dataset/emotion/Asian_Facial_Expression/AsianMovie_0725_0730/list/total.txt"
    image_dir = "/home/PKing/nasdata/dataset/emotion/Asian_Facial_Expression/AsianMovie_0725_0730/images"
    out_dir = "/home/PKing/nasdata/dataset/emotion/Asian_Facial_Expression/dataset-pjq"
    wid = ParseAsian(file, image_dir)
    data_info = wid.parse_data(out_dir=out_dir, max_num=None, vis=False)


if __name__ == '__main__':
    main()
