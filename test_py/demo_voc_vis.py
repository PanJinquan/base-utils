# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-11-29 18:11:47
    @Brief  :
"""
import os
from pybaseutils.dataloader import parser_voc

if __name__ == "__main__":
    names = {'手': 0, '护目镜': 1, '未穿工作服': 2, '身穿工作服': 2, '其他鞋': 3, '绝缘鞋': 3, '安全带': 4, '安全帽': 5,
             '绝缘垫': 6, '绝缘手套': 7, '万用表': 8, '万用表线头': 9, '相序表': 10, '相序表线头': 11, '钳形电流表': 12,
             '电能表': 13, '尖嘴钳': 14, '验电笔': 15, '螺丝刀': 16, '接线盒': 17, '电流互感器': 18, '表箱关': 19,
             '表箱开': 19, '竹梯': 20, '准备区域': 21, '工作台': 22}

    # from models.transforms import data_transforms
    # filename = "/home/dm/nasdata/dataset/csdn/traffic light/VOC/train.txt"
    # class_name = "/home/dm/nasdata/dataset/csdn/traffic light/VOC/class_name.txt"
    # filename = "/home/dm/nasdata/dataset-dmai/handwriting/word-det/word-v1/train.txt"
    # filename = "/home/dm/nasdata/dataset/csdn/face_person/MPII/test.txt"
    # filename = "/home/dm/nasdata/dataset/csdn/Eyeglasses/dataset/face-eyeglasses/trainval.txt"
    # filename = "/home/dm/nasdata/dataset/tmp/insects/VOC2/VOC/VOCdevkit/trainval.txt"
    # filename = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-det-dataset/dataset-v4/train.txt"
    filename = "/home/PKing/nasdata/dataset/tmp/pen/dataset-pen-finger-tip-v2/train/train.txt"
    filename = "/home/PKing/Downloads/S-水表刻度数字检测6800/sample.txt"
    # filename = "/home/dm/nasdata/dataset-dmai/handwriting/word-det/word-old/train.txt"
    # class_name = ["face", "face-eyeglasses"]
    # class_name = "/home/dm/nasdata/dataset/tmp/traffic-sign/TT100K/VOC/train/class_name.txt"
    # class_name = ["unique"]
    class_name =None
    # class_name = ['pen_tip',"finger_tip"]
    dataset = parser_voc.VOCDataset(filename=filename,
                                    data_root=None,
                                    anno_dir=None,
                                    image_dir=None,
                                    class_name=class_name,
                                    transform=None,
                                    use_rgb=False,
                                    check=False,
                                    shuffle=True)
    print("have num:{}".format(len(dataset)))
    class_name = dataset.class_name
    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        image, targets, image_id = data["image"], data["target"], data["image_id"]
        print(image_id)
        bboxes, labels = targets[:, 0:4], targets[:, 4:5]
        parser_voc.show_target_image(image, bboxes, labels, normal=False, transpose=False,
                                     class_name=class_name, use_rgb=False, thickness=3, fontScale=1.2)
        # image = show_boxes_image(image, Dataset.cxcywh2xyxy(bboxes, 0, 0), labels, normal=False, transpose=True)
