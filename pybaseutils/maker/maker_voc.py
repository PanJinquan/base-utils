# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-09-05 19:48:52
    @Brief  :
"""

import cv2
import os
import codecs
from xml.dom.minidom import Document


def write_voc_bboxes_labels(filename, image_shape, bboxes, labels, xmlpath):
    """
    write to xml style of VOC dataset
    :param filename: JPG filename
    :param image_shape: shape=image.shape=[H, W, C]
    :param bboxes: bounding boxes,<list[[tupe]]>=[(xmin,ymin,xmax,ymax),(),...()]
    :param labels: bounding boxes,<list[[tupe]]>=[(xmin,ymin,xmax,ymax),(),...()]
    :param xmlpath:save annotation in *.xml file
    :return: None
    """
    if not image_shape:
        image_shape = [0, 0, 0]
    doc = Document()
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)
    folder = doc.createElement('folder')
    folder_name = doc.createTextNode('widerface')
    folder.appendChild(folder_name)
    annotation.appendChild(folder)

    filenamenode = doc.createElement('filename')
    filename_name = doc.createTextNode(filename)
    filenamenode.appendChild(filename_name)
    annotation.appendChild(filenamenode)

    source = doc.createElement('source')
    annotation.appendChild(source)
    database = doc.createElement('database')
    database.appendChild(doc.createTextNode('wider face Database'))
    source.appendChild(database)

    annotation_s = doc.createElement('annotation')
    annotation_s.appendChild(doc.createTextNode('PASCAL VOC2007'))
    source.appendChild(annotation_s)
    flikerid = doc.createElement('flikerid')
    flikerid.appendChild(doc.createTextNode('-1'))
    source.appendChild(flikerid)

    owner = doc.createElement('owner')
    name_o = doc.createElement('name')
    name_o.appendChild(doc.createTextNode('kinhom'))
    owner.appendChild(name_o)

    size = doc.createElement('size')
    annotation.appendChild(size)
    width = doc.createElement('width')
    width.appendChild(doc.createTextNode(str(image_shape[1])))
    height = doc.createElement('height')
    height.appendChild(doc.createTextNode(str(image_shape[0])))
    depth = doc.createElement('depth')
    depth.appendChild(doc.createTextNode(str(image_shape[2])))
    size.appendChild(width)
    size.appendChild(height)
    size.appendChild(depth)

    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode('0'))
    annotation.appendChild(segmented)

    for i in range(len(bboxes)):
        bbox = bboxes[i]
        name = labels[i]
        objects = doc.createElement('object')
        annotation.appendChild(objects)
        object_name = doc.createElement('name')
        object_name.appendChild(doc.createTextNode(str(name)))
        objects.appendChild(object_name)

        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode('Unspecified'))
        objects.appendChild(pose)

        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('1'))
        objects.appendChild(truncated)

        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        objects.appendChild(difficult)
        bndbox = doc.createElement('bndbox')
        objects.appendChild(bndbox)

        xmin = doc.createElement('xmin')
        xmin.appendChild(doc.createTextNode(str(bbox[0])))
        bndbox.appendChild(xmin)
        ymin = doc.createElement('ymin')
        ymin.appendChild(doc.createTextNode(str(bbox[1])))
        bndbox.appendChild(ymin)
        xmax = doc.createElement('xmax')
        xmax.appendChild(doc.createTextNode(str(bbox[2])))
        bndbox.appendChild(xmax)
        ymax = doc.createElement('ymax')
        ymax.appendChild(doc.createTextNode(str(bbox[3])))
        bndbox.appendChild(ymax)

    f = open(xmlpath, 'w')
    f.write(doc.toprettyxml(indent=' '))
    f.close()


def write_voc_landm_xml_file(filename, image_shape, bboxes, labels, landms, xmlpath):
    """
    write to xml style of VOC dataset
    :param filename: JPG filename
    :param image_shape: shape=image.shape=[H, W, C]
    :param bboxes: bounding boxes,<list[[tupe]]>=[(xmin,ymin,xmax,ymax),(),...()]
    :param landms： [num_bboxes,5,2]
    :param labels: bounding boxes,<list[[tupe]]>=[(xmin,ymin,xmax,ymax),(),...()]
    :param xmlpath:save annotation in *.xml file
    :return: None
    """
    if not image_shape:
        image_shape = [0, 0, 0]
    doc = Document()
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)
    folder = doc.createElement('folder')
    folder_name = doc.createTextNode('widerface')
    folder.appendChild(folder_name)
    annotation.appendChild(folder)

    filenamenode = doc.createElement('filename')
    filename_name = doc.createTextNode(filename)
    filenamenode.appendChild(filename_name)
    annotation.appendChild(filenamenode)

    source = doc.createElement('source')
    annotation.appendChild(source)
    database = doc.createElement('database')
    database.appendChild(doc.createTextNode('wider face Database'))
    source.appendChild(database)

    annotation_s = doc.createElement('annotation')
    annotation_s.appendChild(doc.createTextNode('PASCAL VOC2007'))
    source.appendChild(annotation_s)
    flikerid = doc.createElement('flikerid')
    flikerid.appendChild(doc.createTextNode('-1'))
    source.appendChild(flikerid)

    owner = doc.createElement('owner')
    name_o = doc.createElement('name')
    name_o.appendChild(doc.createTextNode('kinhom'))
    owner.appendChild(name_o)

    size = doc.createElement('size')
    annotation.appendChild(size)
    width = doc.createElement('width')
    width.appendChild(doc.createTextNode(str(image_shape[1])))
    height = doc.createElement('height')
    height.appendChild(doc.createTextNode(str(image_shape[0])))
    depth = doc.createElement('depth')
    depth.appendChild(doc.createTextNode(str(image_shape[2])))
    size.appendChild(width)
    size.appendChild(height)
    size.appendChild(depth)

    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode('0'))
    annotation.appendChild(segmented)

    for i in range(len(bboxes)):
        bbox = bboxes[i]
        landm = landms[i]
        name = labels[i]
        objects = doc.createElement('object')
        annotation.appendChild(objects)
        object_name = doc.createElement('name')
        object_name.appendChild(doc.createTextNode(str(name)))
        objects.appendChild(object_name)

        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode('Unspecified'))
        objects.appendChild(pose)

        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('1'))
        objects.appendChild(truncated)

        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        objects.appendChild(difficult)

        bndbox = doc.createElement('bndbox')
        objects.appendChild(bndbox)
        xmin = doc.createElement('xmin')
        xmin.appendChild(doc.createTextNode(str(bbox[0])))
        bndbox.appendChild(xmin)
        ymin = doc.createElement('ymin')
        ymin.appendChild(doc.createTextNode(str(bbox[1])))
        bndbox.appendChild(ymin)
        xmax = doc.createElement('xmax')
        xmax.appendChild(doc.createTextNode(str(bbox[2])))
        bndbox.appendChild(xmax)
        ymax = doc.createElement('ymax')
        ymax.appendChild(doc.createTextNode(str(bbox[3])))
        bndbox.appendChild(ymax)

        if len(landm) > 0:
            lm = doc.createElement('lm')
            objects.appendChild(lm)
            #
            x = doc.createElement('x1')
            x.appendChild(doc.createTextNode(str(landm[0][0])))
            lm.appendChild(x)
            y = doc.createElement('y1')
            y.appendChild(doc.createTextNode(str(landm[0][1])))
            lm.appendChild(y)

            x = doc.createElement('x2')
            x.appendChild(doc.createTextNode(str(landm[1][0])))
            lm.appendChild(x)
            y = doc.createElement('y2')
            y.appendChild(doc.createTextNode(str(landm[1][1])))
            lm.appendChild(y)

            x = doc.createElement('x3')
            x.appendChild(doc.createTextNode(str(landm[2][0])))
            lm.appendChild(x)
            y = doc.createElement('y3')
            y.appendChild(doc.createTextNode(str(landm[2][1])))
            lm.appendChild(y)

            x = doc.createElement('x4')
            x.appendChild(doc.createTextNode(str(landm[3][0])))
            lm.appendChild(x)
            y = doc.createElement('y4')
            y.appendChild(doc.createTextNode(str(landm[3][1])))
            lm.appendChild(y)

            x = doc.createElement('x5')
            x.appendChild(doc.createTextNode(str(landm[4][0])))
            lm.appendChild(x)
            y = doc.createElement('y5')
            y.appendChild(doc.createTextNode(str(landm[4][1])))
            lm.appendChild(y)

            v = doc.createElement('visible')
            v.appendChild(doc.createTextNode(str(1)))
            lm.appendChild(v)
            blur = doc.createElement('blur')
            blur.appendChild(doc.createTextNode(str(0.28)))
            lm.appendChild(blur)

    f = open(xmlpath, 'w')
    f.write(doc.toprettyxml(indent=' '))
    f.close()


def write_voc_landm_xml_objects(filename, image_shape, objects: list, xml_path):
    """
    write_voc_landm_xml_file(filename, image_shape, bboxes, labels, landms, xmlpath):
    :param image_shape:image_dict.shape
    :param filename: file name
    :param xml_path: save Annotations(*.xml) file path
    :param objects: [object] ,object= {
                                    "name": name,
                                    "bndbox": bndbox,
                                    "keypoints": keypoint
                                     }
            - name: bbox label name
            - bndbox: bbox =[x_min, y_min, x_max, y_max]
            - keypoint: [x_1, y_1, v_1,...,x_k, y_k, v_k],
                    其中x,y为Keypoint的坐标，v为可见标志
                    v = 0 : 未标注点
                    v = 1 : 标注了但是图像中不可见（例如遮挡）
                    v = 2 : 标注了并图像可见
    :return:
    """
    height, width, depth = image_shape
    xml = codecs.open(xml_path, 'w', encoding='utf-8')
    xml.write('<annotation>\n')
    xml.write('\t<folder>' + 'VOC2007' + '</folder>\n')
    xml.write('\t<filename>' + filename + '</filename>\n')
    xml.write('\t<source>\n')
    xml.write('\t\t<database>The VOC2007 Database</database>\n')
    xml.write('\t\t<annotation>PASCAL VOC2007</annotation>\n')
    xml.write('\t\t<image_dict>flickr</image_dict>\n')
    xml.write('\t\t<flickrid>NULL</flickrid>\n')
    xml.write('\t</source>\n')
    xml.write('\t<owner>\n')
    xml.write('\t\t<flickrid>NULL</flickrid>\n')
    xml.write('\t\t<name>pjq</name>\n')
    xml.write('\t</owner>\n')
    xml.write('\t<size>\n')
    xml.write('\t\t<width>' + str(width) + '</width>\n')
    xml.write('\t\t<height>' + str(height) + '</height>\n')
    xml.write('\t\t<depth>' + str(depth) + '</depth>\n')
    xml.write('\t</size>\n')
    xml.write('\t\t<segmented>0</segmented>\n')
    for obj in objects:
        name = obj["name"]
        x_min, y_min, x_max, y_max = obj["bndbox"]
        xml.write('\t<object>\n')
        xml.write('\t\t<name>{}</name>\n'.format(name))
        xml.write('\t\t<pose>Unspecified</pose>\n')
        xml.write('\t\t<truncated>0</truncated>\n')
        xml.write('\t\t<difficult>0</difficult>\n')
        xml.write('\t\t<bndbox>\n')
        xml.write('\t\t\t<xmin>' + str(x_min) + '</xmin>\n')
        xml.write('\t\t\t<ymin>' + str(y_min) + '</ymin>\n')
        xml.write('\t\t\t<xmax>' + str(x_max) + '</xmax>\n')
        xml.write('\t\t\t<ymax>' + str(y_max) + '</ymax>\n')
        xml.write('\t\t</bndbox>\n')
        if "keypoints" in obj:
            lm = obj["keypoints"]
            xml.write('\t\t<lm>\n')
            visible = 1
            for i in range(len(lm)):
                x, y = lm[i][0], lm[i][1]
                xml.write('\t\t\t<x{}>'.format(i + 1) + str(x) + '</x{}>\n'.format(i + 1))
                xml.write('\t\t\t<y{}>'.format(i + 1) + str(y) + '</y{}>\n'.format(i + 1))
                if x < 0 or x > width or y < 0 or y > height:
                    visible = 0
            xml.write('\t\t\t<visible>' + str(visible) + '</visible>\n')
            xml.write('\t\t</lm>\n')
        xml.write('\t</object>\n')
    xml.write('</annotation>')


def write_voc_xml_objects(filename, image_shape, objects: list, xml_path):
    """
    write_voc_landm_xml_file(filename, image_shape, bboxes, labels, landms, xmlpath):
    :param image_shape:image_dict.shape
    :param filename: file name
    :param xml_path: save Annotations(*.xml) file path
    :param objects: [object] ,object= {
                                    "name": name,
                                    "bndbox": bndbox,
                                    "keypoints": keypoint
                                     }
            - name: bbox label name
            - bndbox: bbox =[x_min, y_min, x_max, y_max]
            - keypoint: [x_1, y_1, v_1,...,x_k, y_k, v_k],
                    其中x,y为Keypoint的坐标，v为可见标志
                    v = 0 : 未标注点
                    v = 1 : 标注了但是图像中不可见（例如遮挡）
                    v = 2 : 标注了并图像可见
    :return:
    """
    height, width, depth = image_shape
    xml = codecs.open(xml_path, 'w', encoding='utf-8')
    xml.write('<annotation>\n')
    xml.write('\t<folder>' + 'VOC2007' + '</folder>\n')
    xml.write('\t<filename>' + filename + '</filename>\n')
    xml.write('\t<source>\n')
    xml.write('\t\t<database>The VOC2007 Database</database>\n')
    xml.write('\t\t<annotation>PASCAL VOC2007</annotation>\n')
    xml.write('\t\t<image_dict>flickr</image_dict>\n')
    xml.write('\t\t<flickrid>NULL</flickrid>\n')
    xml.write('\t</source>\n')
    xml.write('\t<owner>\n')
    xml.write('\t\t<flickrid>NULL</flickrid>\n')
    xml.write('\t\t<name>pjq</name>\n')
    xml.write('\t</owner>\n')
    xml.write('\t<size>\n')
    xml.write('\t\t<width>' + str(width) + '</width>\n')
    xml.write('\t\t<height>' + str(height) + '</height>\n')
    xml.write('\t\t<depth>' + str(depth) + '</depth>\n')
    xml.write('\t</size>\n')
    xml.write('\t\t<segmented>0</segmented>\n')
    for obj in objects:
        name = obj["name"]
        x_min, y_min, x_max, y_max = obj["bndbox"]
        xml.write('\t<object>\n')
        xml.write('\t\t<name>{}</name>\n'.format(name))
        xml.write('\t\t<pose>Unspecified</pose>\n')
        xml.write('\t\t<truncated>0</truncated>\n')
        xml.write('\t\t<difficult>0</difficult>\n')
        xml.write('\t\t<bndbox>\n')
        xml.write('\t\t\t<xmin>' + str(x_min) + '</xmin>\n')
        xml.write('\t\t\t<ymin>' + str(y_min) + '</ymin>\n')
        xml.write('\t\t\t<xmax>' + str(x_max) + '</xmax>\n')
        xml.write('\t\t\t<ymax>' + str(y_max) + '</ymax>\n')
        xml.write('\t\t</bndbox>\n')
        if "keypoints" in obj:
            keypoints = obj["keypoints"]
            add_keypoints(xml, keypoints)
        xml.write('\t</object>\n')
    xml.write('</annotation>')


def add_keypoints(xml, keypoint: list):
    keypoint = [str(i) for i in keypoint]
    keypoint = ",".join(keypoint)
    xml.write('\t\t<keypoints>{}</keypoints>\n'.format(keypoint))


def create_object(name, bndbox, keypoint=[]):
    if keypoint:
        object = {"name": name, "bndbox": bndbox, "keypoints": keypoint}
    else:
        object = {"name": name, "bndbox": bndbox}
    return object


def create_objects(bboxes: list, labels: list, keypoints=None, class_name={}):
    """
    :param bboxes: [[xmin,ymin,xmax,ymax]]
    :param labels:[name1,name2]
    :param keypoint:
    :return:
    """
    object = []
    for i in range(len(labels)):
        name = class_name[labels[i]] if class_name else labels[i]
        bndbox = bboxes[i]
        keypoint = keypoints[i] if keypoints else []
        obj = create_object(name, bndbox, keypoint)
        object.append(obj)
    return object


def create_voc_demo(image_path, out_anno_dir):
    image = cv2.imread(image_path)
    image_shape = image.shape
    filename = os.path.basename(image_path)
    id = filename[:-len(".jpg")]
    xml_path = os.path.join(out_anno_dir, "{}.xml".format(id))
    keypoint = [100, 200, 2, 510, 191, 2, 506, 191, 2, 512, 192, 2, 503, 192, 1, 515, 202, 2,
                499, 202, 2, 524, 214, 2, 497, 215, 2, 516, 226, 2, 496, 224, 2, 511, 232, 2,
                497, 230, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    keypoint = []
    object1 = create_object(name="dog", bndbox=[48, 240, 195, 371], keypoint=keypoint)
    object2 = create_object(name="person", bndbox=[8, 12, 352, 498], keypoint=keypoint)
    objects = []
    objects.append(object1)
    objects.append(object2)
    write_voc_xml_objects(image_shape, filename, xml_path, objects)


if __name__ == "__main__":
    label = 1
    out_anno_dir = "/media/dm/dm1/git/python-learning-notes/dataset/VOC/Annotations"
    image_path = "/media/dm/dm1/git/python-learning-notes/dataset/VOC/JPEGImages/000000.jpg"
    create_voc_demo(image_path, out_anno_dir)
