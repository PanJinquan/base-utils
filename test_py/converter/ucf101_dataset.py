# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-10-12 16:21:55
    @Brief  :
"""
import os
import cv2
from tqdm import tqdm
from pybaseutils import file_utils, image_utils


def ucf101_dataset(filename, video_dir, out_dir):
    file_list = file_utils.read_data(filename, split=" ")
    print(filename, len(file_list))
    print(out_dir)
    for data in tqdm(file_list):
        name = data[0]
        src_file = os.path.join(video_dir, name)
        dst_file = os.path.join(out_dir, name)
        # file_utils.copy_file(src_file, dst_file)
        file_utils.move_file(src_file, dst_file)


if __name__ == "__main__":
    video_dir = "/home/PKing/nasdata/dataset/action/UCU-101/UCF101/UCF-101"
    # TODO UCF101数据集按照有三种数据集划分方法，默认使用(trainlist01.txt,testlist01.txt)划分方式，
    train = "/home/PKing/nasdata/dataset/action/UCU-101/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/trainlist01.txt"
    test = "/home/PKing/nasdata/dataset/action/UCU-101/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/testlist01.txt"
    # TODO 划分训练集
    out_dir = os.path.join(os.path.dirname(video_dir), "train")
    ucf101_dataset(train, video_dir, out_dir)
    # TODO 划分测试集
    out_dir = os.path.join(os.path.dirname(video_dir), "test")
    ucf101_dataset(test, video_dir, out_dir)
