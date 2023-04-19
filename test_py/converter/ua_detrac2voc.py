# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-10-12 16:21:55
    @Brief  :
"""
import os
import cv2
import xmltodict
from tqdm import tqdm
from pybaseutils.maker import maker_voc
from pybaseutils import file_utils, image_utils


def read_xml2json(xml_file):
    """
    import xmltodict
    :param xml_file:
    :return:
    """
    with open(xml_file, encoding='utf-8') as fd:  # 将XML文件装载到dict里面
        content = xmltodict.parse(fd.read())
    return content


def parser_annotations(objects):
    """
    解析标注信息
    """
    rects = []
    labels = []
    targets = objects['target_list']['target']  # collections.OrderedDict
    if not isinstance(targets, list): targets = [targets]
    for data in targets:
        box = data['box']
        attribute = data['attribute']
        label = attribute['@vehicle_type']
        rect = [box['@left'], box['@top'], box['@width'], box['@height']]
        rect = [float(r) for r in rect]
        rects.append(rect)
        labels.append(label)
    bboxes = image_utils.rects2bboxes(rects)
    num = objects['@num']
    image_id = "img{:0=5d}.jpg".format(int(num))  # img00002.jpg
    return image_id, bboxes, labels


def get_ignored_region(objects):
    """获得ignored区域"""
    rects = []
    labels = []
    if not "box" in objects: return rects, labels
    for data in objects["box"]:
        label = "ignored_region"
        rect = [data['@left'], data['@top'], data['@width'], data['@height']]
        rect = [float(r) for r in rect]
        rects.append(rect)
        labels.append(label)
    bboxes = image_utils.rects2bboxes(rects)
    return bboxes, labels


def show_ua_detrac_dataset(image_dir, annot_dir, out_draw="", vis=False):
    """
    可视化车辆检测数据集
    class_set:['car', 'bus', 'others', 'van']
    :param image_dir: UA-DETRAC数据集图片(*.jpg)根目录
    :param annot_dir:  UA-DETRAC数据集标注文件(*.xml)根目录
    :param vis: 是否可视化效果
    """
    print("image_dir:{}".format(image_dir))
    print("annot_dir:{}".format(annot_dir))
    xml_list = file_utils.get_files_list(annot_dir, postfix=["*.xml"])
    class_set = []
    for annot_file in tqdm(xml_list):
        print(annot_file)
        # 将xml转换为OrderedDict格式，方便解析
        annotations = read_xml2json(annot_file)
        subname = annotations['sequence']['@name']  # UA-DETRAC子目录
        # 被忽略的区域
        ignore_bboxes, ignore_labels = get_ignored_region(annotations['sequence']['ignored_region'])
        # 遍一帧图像，获得车辆目标框
        frame_info = annotations['sequence']['frame']
        for i in range(len(frame_info)):
            image_name, bboxes, labels = parser_annotations(frame_info[i])
            image_id = image_name.split(".")[0]
            image_file = os.path.join(image_dir, subname, image_name)
            class_set = labels + class_set
            class_set = list(set(class_set))
            if not os.path.exists(image_file):
                print("not exist:{}".format(image_file))
                continue
            image = cv2.imread(image_file)
            image = image_utils.draw_image_bboxes_text(image, ignore_bboxes, ignore_labels,
                                                       color=(10, 10, 10), thickness=2, fontScale=1.0)
            image = image_utils.draw_image_bboxes_text(image, bboxes, labels,
                                                       color=(255, 0, 0), thickness=2, fontScale=1.0)
            if out_draw:
                dst_file = file_utils.create_dir(out_draw, None, "{}_{}.jpg".format(subname, image_id))
                cv2.imwrite(dst_file, image)
            if vis:
                image_utils.cv_show_image("det", image, use_rgb=False)
    print("class_set:{}".format(class_set))


def converter_ua_detrac2voc(image_dir, annot_dir, out_voc, vis=True):
    """
    将车辆检测数据集UA-DETRAC转换为VOC数据格式(xmin,ymin,xmax,ymax)
    class_set:['car', 'bus', 'others', 'van']
    :param image_dir: UA-DETRAC数据集图片(*.jpg)根目录
    :param annot_dir:  UA-DETRAC数据集标注文件(*.xml)根目录
    :param out_voc: 输出VOC格式数据集目录
    :param vis: 是否可视化效果
    """
    print("image_dir:{}".format(image_dir))
    print("annot_dir:{}".format(annot_dir))
    print("out_voc  :{}".format(out_voc))
    xml_list = file_utils.get_files_list(annot_dir, postfix=["*.xml"])
    out_image_dir = file_utils.create_dir(out_voc, None, "JPEGImages")
    out_xml_dir = file_utils.create_dir(out_voc, None, "Annotations")
    class_set = []
    for annot_file in tqdm(xml_list):
        print(annot_file)
        # 将xml转换为OrderedDict格式，方便解析
        annotations = read_xml2json(annot_file)
        subname = annotations['sequence']['@name']  # UA-DETRAC子目录
        # 被忽略的区域
        ignore_bboxes, ignore_labels = get_ignored_region(annotations['sequence']['ignored_region'])
        # 遍一帧图像，获得车辆目标框
        frame_info = annotations['sequence']['frame']
        for i in range(len(frame_info)):
            image_name, bboxes, labels = parser_annotations(frame_info[i])
            image_id = image_name.split(".")[0]
            image_file = os.path.join(image_dir, subname, image_name)
            class_set = labels + class_set
            class_set = list(set(class_set))
            if not os.path.exists(image_file):
                print("not exist:{}".format(image_file))
                continue
            image = cv2.imread(image_file)
            image_shape = image.shape
            new_image_id = "{}_{}".format(subname, image_id)
            new_name = "{}.jpg".format(new_image_id)
            xml_path = file_utils.create_dir(out_xml_dir, None, "{}.xml".format(new_image_id))
            objects = maker_voc.create_objects(bboxes, labels)
            maker_voc.write_voc_xml_objects(new_name, image_shape, objects, xml_path)
            dst_file = file_utils.create_dir(out_image_dir, None, new_name)
            file_utils.copy_file(image_file, dst_file)
            # cv2.imwrite(dst_file, image)
            if vis:
                image = image_utils.draw_image_bboxes_text(image, ignore_bboxes, ignore_labels,
                                                           color=(10, 10, 10), thickness=2, fontScale=1.0)
                image = image_utils.draw_image_bboxes_text(image, bboxes, labels,
                                                           color=(255, 0, 0), thickness=2, fontScale=1.0)
                image_utils.cv_show_image("det", image, use_rgb=False)
    file_utils.save_file_list(out_image_dir, filename=None, prefix="", postfix=file_utils.IMG_POSTFIX,
                              only_id=False, shuffle=False, max_num=None)
    print("class_set:{}".format(class_set))


if __name__ == "__main__":
    """
    pip install pybaseutils
    pip install xmltodict
    """
    image_dir = "/home/dm/nasdata/dataset/csdn/car/UA-DETRAC/DETRAC-train-data/Insight-MVT_Annotation_Train"
    annot_dir = "/home/dm/nasdata/dataset/csdn/car/UA-DETRAC/DETRAC-Train-Annotations-XML"
    # 可视化车辆检测数据集
    out_draw = os.path.join(os.path.dirname(image_dir), "result")
    show_ua_detrac_dataset(image_dir, annot_dir, out_draw=out_draw, vis=True)

    # 将车辆检测数据集UA - DETRAC转换为VOC数据格式
    out_voc = os.path.join(os.path.dirname(image_dir), "VOC")
    converter_ua_detrac2voc(image_dir, annot_dir, out_voc, vis=True)
