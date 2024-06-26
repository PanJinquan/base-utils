# -*- coding: utf-8 -*-

import os
import sys

sys.path.insert(0, "/home/dm/nasdata/release/detector/object-detector")
import random
import cv2
import numpy as np
from tqdm import tqdm
from pybaseutils import file_utils, image_utils, pandas_utils
from libs.detector.detector import Detector

class_name = {0: "neutral",
              1: "happy",
              2: "excited",
              3: "surprised",
              4: "sad",
              5: "disgust",
              6: "fear",
              7: "confused",
              8: "tired",
              9: "angry",
              10: "serious"
              }


class ParseAsian(object):
    def __init__(self, image_dir, file, shuffle=True, prb_thresh=0.9, iou_thresh=0.3):
        """
        0: Neutral
        1: Happy
        2: Excited
        3: Surprised
        4: Sad
        5:
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
        # self.detector = Detector(prb_thresh=prb_thresh, iou_thresh=iou_thresh, detect_type="mtcnn")
        self.detector = Detector(prb_thresh=prb_thresh, iou_thresh=iou_thresh, detect_type="dfsd")
        self.image_dir = image_dir
        self.shuffle = shuffle
        if file:
            data = pandas_utils.read_csv(file)
            self.item_list = pandas_utils.get_rows_by_keys(data, keys=["pth", "label"]).values.tolist()
        else:
            images = file_utils.get_images_list(image_dir)
            images = file_utils.get_sub_list(images, image_dir)
            self.item_list = [[path, os.path.dirname(path)] for path in images]
        if shuffle:
            random.seed(200)
            random.shuffle(self.item_list)

    def parse_data(self, out_dir="", flag="20220202", max_num=None, vis=True):
        nums_sample = len(self.item_list)
        if max_num:
            nums_sample = min(max_num, nums_sample)
        dst_item_list = []
        for i in tqdm(range(nums_sample)):
            image_file, label = self.item_list[i]
            image_path = os.path.join(self.image_dir, image_file)
            if not os.path.exists(image_path):
                print("no path:{}".format(image_path))
                continue

            image = image_utils.read_image(image_path)
            h, w = image.shape[:2]
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            dets, _ = self.detector.detect(rgb, vis=False)
            if len(dets) == 0:
                boxes = [[0, 0, w, h]]
                boxes = image_utils.extend_xyxy(boxes, scale=[0.7, 0.9])
            else:
                boxes = dets[:, 0:4]
            xmin, ymin, xmax, ymax = np.asarray(boxes, dtype=np.int32)[0]
            sub, name = image_file.split("/")
            name = "{}_{:0=6d}.jpg".format(flag, i)
            # item = [os.path.join(label, name), xmin, ymin, xmax, ymax]
            item = [image_file, label, xmin, ymin, xmax, ymax]
            box = [xmin, ymin, xmax, ymax]
            dst_item_list.append(item)
            if out_dir:
                # orig_file = file_utils.create_dir(os.path.join(out_dir, "origin"), label, name)
                # file_utils.copy_file(image_path, orig_file)
                crop_file = file_utils.create_dir(os.path.join(out_dir, "crops"), label, name)
                crop = image_utils.get_box_crop(image, box)
                cv2.imwrite(crop_file, crop)
            if vis:
                image = image_utils.draw_image_boxes(image, [box])
                image = image_utils.draw_texts(image, [[xmin + 5, ymin + 10]], texts=[label])
                image_utils.cv_show_image("image", image)
        if not out_dir: out_dir = self.image_dir
        filename = file_utils.create_dir(os.path.join(out_dir, "origin"), None, "label.txt")
        file_utils.write_data(filename, dst_item_list, split=",")
        return dst_item_list


def main():
    image_dir = "/home/dm/nasdata/dataset/csdn/emotion-dataset/AffectNet/origin"
    out_dir = "/home/dm/nasdata/dataset/csdn/emotion-dataset/AffectNet"
    wid = ParseAsian(image_dir, "")
    data_info = wid.parse_data(out_dir=out_dir, max_num=None, vis=False)


if __name__ == '__main__':
    main()
