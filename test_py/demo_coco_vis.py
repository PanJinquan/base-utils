# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-11-29 18:11:47
    @Brief  :
"""
import os
import numpy as np
from pybaseutils import image_utils, file_utils
from pybaseutils.dataloader import parser_coco_ins, parser_coco_kps, parser_coco_det


def demo_vis_CocoInstances():
    anno_file2 = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-outdoor-det/dataset-v3/train-coco.json"
    names = {'手': 0, '护目镜': 1, '未穿工作服': 2, '身穿工作服': 2, '其他鞋': 3, '绝缘鞋': 3, '安全带': 4, '安全帽': 5,
             '绝缘垫': 6, '绝缘手套': 7, '万用表': 8, '万用表线头': 9, '相序表': 10, '相序表线头': 11, '钳形电流表': 12,
             '电能表': 13, '尖嘴钳': 14, '验电笔': 15, '螺丝刀': 16, '接线盒': 17, '电流互感器': 18, '表箱关': 19,
             '表箱开': 19, '竹梯': 20, '准备区域': 21, '工作台': 22}

    # class_name = {'手': 0, '护目镜': 1, '未穿工作服': 2, '身穿工作服': 2, '其他鞋': 3, '绝缘鞋': 3, '安全带': 4, '安全帽': 5,
    #               '绝缘垫': 6, '绝缘手套': 7, '万用表': 8, '万用表线头': 9, '相序表': 10, '相序表线头': 11, '钳形电流表': 12,
    #               '电能表': 13, '尖嘴钳': 14, '验电笔': 15, '螺丝刀': 16, '接线盒': 17, '电流互感器': 18, '表箱关': 19,
    #               '表箱开': 19, '竹梯': 20, '准备区域': 21, '工作台': 22}
    class_name = ["BG", '遮拦杆', '主杆', '垫子', '柱式绝缘子', '抹布', '吊物绳', '绝缘手套', '脚扣', '安全帽', '导线头', '尖嘴钳',
                  '扳手', '螺丝', '铁架', '身穿工作服', '安全带', '绝缘鞋', '工具袋', '铝扎线', '止步高压危险标示牌',
                  '从此进出标示牌', '在此工作标示牌', '手', '其他鞋', '安全绳', '未穿工作服']

    class_name = ['手', '未穿工作服', '身穿工作服', '其他鞋', '绝缘鞋', '安全带', '安全帽', '安全绳', '垫子', '绝缘手套',
                  '主杆', '柱式绝缘子', '抹布', '吊物绳', '脚扣', '尖嘴钳', '扳手', '螺丝', '铁架', '工具袋', '铝扎线', '导线头',
                  '遮拦杆', '止步高压危险标示牌', '从此进出标示牌', '在此工作标示牌', ]
    class_name = ["尖嘴钳"]

    anno_file2 = "/media/PKing/新加卷1/SDK/base-utils/data/coco/coco_ins.json"
    class_name = None
    dataset = parser_coco_ins.CocoInstances(anno_file=[anno_file2], image_dir="",
                                            class_name=class_name, use_rgb=False, shuffle=True)
    class_name = dataset.class_name
    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        image, boxes, labels, mask = data['image'], data["boxes"], data["labels"], data["mask"]
        segs = data["segs"]
        points = [image_utils.find_minAreaRect(seg)[0] for seg in segs]
        print("i={},image_file={}".format(i, data["image_file"]))
        # dataset.showAnns(image, data['annotations'])
        parser_coco_ins.show_target_image(image, mask, boxes, labels, points=points,class_name=class_name)


def demo_vis_CocoKeypoints():
    # anno_file = "/home/PKing/nasdata/dataset/tmp/hand-pose/HandPose-v2/train/train_anno.json"
    # class_name = ['hand']
    # anno_file = "/home/PKing/nasdata/dataset/tmp/pen/dataset-pen2/train/coco_kps.json"
    anno_file = "/home/PKing/nasdata/tmp/tmp/pen/dataset-pen1/train/coco_kps.json"
    # class_name = ['pen_tip']
    class_name = []
    class_name=["hand_pen"]
    # class_name = {"pen": "hand"}
    dataset = parser_coco_kps.CocoKeypoints(anno_file, image_dir="", class_name=class_name, shuffle=True)
    bones = dataset.bones
    for i in range(len(dataset)):
        # i=4
        data = dataset.__getitem__(i)
        image, boxes, labels, keypoints = data['image'], data["boxes"], data["labels"], data["keypoints"]
        h,w = image.shape[:2]
        kpts = [s / (w, h) for s in keypoints] if len(keypoints) > 0 else []
        kpts = np.asarray(kpts)
        keypoints = np.asarray(keypoints)
        print("i={},image_file={}".format(i, data['image_file']))
        parser_coco_kps.show_target_image(image, keypoints, boxes, colors=bones["colors"],
                                          skeleton=bones["skeleton"], thickness=4)


def demo_vis_CocoDetections():
    # hand
    anno_file = "/home/PKing/nasdata/dataset/tmp/hand-pose/HandPose-v1/test/test_anno.json"
    class_name = []
    dataset = parser_coco_det.CocoDetections(anno_file, image_dir="", class_name=class_name)
    class_name = dataset.class_name
    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        image, targets, image_id = data['image'], data["target"], data["image_id"]
        bboxes, labels = targets[:, 0:4], targets[:, 4:5]
        print("i={},image_id={}".format(i, data["image_id"]))
        parser_coco_det.show_target_image(image, bboxes, labels, normal=False, transpose=False, class_name=class_name)


if __name__ == "__main__":
    # demo_vis_CocoInstances()
    demo_vis_CocoKeypoints()
    # demo_vis_CocoDetections()
