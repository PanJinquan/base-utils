# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-11-29 18:49:56
    @Brief  : https://blog.csdn.net/qq_38253797/article/details/125042833
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from pybaseutils.converter import build_voc
from pybaseutils import file_utils, image_utils, yaml_utils


def get_plate_licenses(plate):
    """
    普通蓝牌共有7位字符；新能源车牌有8位字符： https://baike.baidu.com/item/%E8%BD%A6%E7%89%8C/8347320?fr=aladdin
    《新能源电动汽车牌照和普通牌照区别介绍》https://www.yoojia.com/ask/4-11906976349117851507.html
    新能源汽车车牌可分为三部分：省份简称(1位汉字)十地方行政区代号(1位字母)十序号(6位)
    字母“D”代表纯电动汽车；
    字母“F”代表非纯电动汽车(包括插电式混合动力和燃料电池汽车等)。
    :param plate:
    :return:
    """
    provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
                 "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
    alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                 'W', 'X', 'Y', 'Z', 'O']
    ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
           'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
    result = [provinces[int(plate[0])], alphabets[int(plate[1])]]
    result += [ads[int(p)] for p in plate[2:]]
    result = "".join(result)
    # 新能源车牌的要求，如果不是新能源车牌可以删掉这个if
    # if result[2] != 'D' and result[2] != 'F' \
    #         and result[-1] != 'D' and result[-1] != 'F':
    #     print(plate)
    #     print("Error label, Please check!")
    print(plate, result)
    return result


def parser_annotations(image_file):
    """
    :param image_file: 图片路径
    :return: 返回标注信息info
    """
    filename = os.path.basename(image_file)
    try:
        annotations = filename.split("-")
        rate = annotations[0]  # 车牌区域占整个画面的比例；
        angle = annotations[1].split("_")  # 车牌水平和垂直角度, 水平95°, 竖直113°
        box = annotations[2].replace("&", "_").split("_")  # 标注框左上、右下坐标，左上(154, 383), 右下(386, 473)
        point = annotations[3].replace("&", "_").split("_")  # 标注框四个角点坐标，顺序为右下、左下、左上、右上
        plate = annotations[4].split("_")  # licenses 标注框四个角点坐标，顺序为右下、左下、左上、右上
        plate = get_plate_licenses(plate)
        box = [int(b) for b in box]
        point = [int(b) for b in point]
        point = np.asarray(point).reshape(-1, 2)
        bboxes = [box]
        angles = [angle]
        points = [point]
        plates = [plate]
        labels = ["plate"] * len(bboxes)
    except Exception as e:
        bboxes = []
        points = []
        labels = []
        plates = []
        angles = []
    info = {"filename": filename, "bboxes": bboxes, "points": points,
            "labels": labels, "plates": plates, "angles": angles}
    return info


def save_plate_licenses(image, bboxes, plates, out_dir, name=""):
    crops = image_utils.get_bboxes_crop(image, bboxes)
    for i in range(len(crops)):
        label = plates[i]
        # image_ids = file_utils.get_time(format="p")
        file = os.path.join(out_dir, "{}_{}_{:0=3d}.jpg".format(label, name, i))
        file_utils.create_file_path(file)
        cv2.imwrite(file, crops[i])


def converter_CCPD2voc(image_dir, out_voc, vis=True):
    """
    将CCPD数据集转换为VOC数据格式(xmin,ymin,xmax,ymax)
    :param image_dir: BITVehicle数据集图片(*.jpg)根目录
    :param annot_file: BITVehicle数据集标注文件VehicleInfo.mat
    :param out_voc: 输出VOC格式数据集目录
    :param vis: 是否可视化效果
    """
    print("image_dir :{}".format(image_dir))
    print("out_voc   :{}".format(out_voc))
    out_image_dir = file_utils.create_dir(out_voc, None, "JPEGImages")
    out_xml_dir = file_utils.create_dir(out_voc, None, "Annotations")
    out_crop_dir = file_utils.create_dir(out_voc, None, "plates")
    class_set = []
    image_list = file_utils.get_images_list(image_dir)
    for i, image_file in enumerate(tqdm(image_list)):
        info = parser_annotations(image_file)
        labels = info["labels"]
        bboxes = info["bboxes"]
        points = info["points"]
        plates = info["plates"]
        angles = info["angles"]
        image_name = info["filename"]
        print("i={},plates:{},angles(水平,垂直角度):{}".format(os.path.basename(image_file), plates, angles))
        if len(labels) == 0:
            continue
        image_name = os.path.basename(image_name)
        img_postfix = image_name.split(".")[-1]
        image_id = image_name[:-len(img_postfix) - 1]

        class_set = labels + class_set
        class_set = list(set(class_set))
        if not os.path.exists(image_file):
            print("not exist:{}".format(image_file))
            continue
        image = cv2.imread(image_file)
        name = "{:0=5d}".format(i)
        save_plate_licenses(image, bboxes, plates, out_dir=out_crop_dir, name=name)
        image_shape = image.shape
        xml_path = file_utils.create_dir(out_xml_dir, None, "{}.xml".format(image_id))
        dst_file = file_utils.create_dir(out_image_dir, None, "{}.{}".format(image_id, img_postfix))
        objects = build_voc.create_objects(bboxes, labels)
        # XML不支持&的文件明，需要将&替换为#
        build_voc.write_voc_xml_objects(image_name.replace("&", "#"), image_shape, objects, xml_path)
        file_utils.copy_file(image_file, dst_file)
        cv2.imwrite(dst_file, image)
        if vis:
            image = image_utils.draw_image_bboxes_text(image, bboxes, plates, color=(255, 0, 0), thickness=2,
                                                       fontScale=0.8, drawType="chinese")
            # image = image_utils.draw_image_points_lines(image, points=points[0], line_color=(0, 0, 255))
            image_utils.cv_show_image("det", image, use_rgb=False, delay=0)
    file_utils.save_file_list(out_image_dir, filename=None, prefix="", postfix=file_utils.IMG_POSTFIX,
                              only_id=False, shuffle=False, max_num=None)
    print("class_set:{}".format(class_set))


if __name__ == "__main__":
    dataset = "/home/dm/nasdata/dataset/csdn/plate/dataset/CCPD2019"
    output = "/home/dm/nasdata/dataset/csdn/plate/dataset/CCPD2019-voc"
    sub_list = file_utils.get_sub_paths(dataset)
    for sub in sub_list:
        out_voc = os.path.join(output, sub)
        image_dir = os.path.join(dataset, sub)
        converter_CCPD2voc(image_dir, out_voc, vis=False)
