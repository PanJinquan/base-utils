# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-11-29 18:11:47
    @Brief  :
"""
import os
import cv2
from tqdm import tqdm
from pybaseutils.dataloader import parser_voc
from pybaseutils import image_utils, file_utils


def save_object_crops(image, out_dir, bboxes, labels, image_id, class_name=None,
                      scale=[], square=False, padding=False, flag="", vis=False):
    """
    对VOC的数据目标进行裁剪
    :param image:
    :param out_dir:
    :param bboxes:
    :param labels:
    :param image_id:
    :param class_name:
    :param scale:
    :param square:
    :param padding:
    :param flag:
    :param vis:
    :return:
    """
    image_id = image_id.split(".")[0]
    if square:
        bboxes = image_utils.get_square_boxes(bboxes, use_max=True, baseline=-1)
    if scale:
        bboxes = image_utils.extend_xyxy(bboxes, scale=scale)
    if padding:
        crops = image_utils.get_bboxes_crop_padding(image, bboxes)
    else:
        crops = image_utils.get_bboxes_crop(image, bboxes)
    if vis:
        m = image_utils.draw_image_bboxes_labels(image.copy(), bboxes, labels, class_name=class_name,
                                                 thickness=2, fontScale=0.8, drawType="custom")
        image_utils.cv_show_image("image", m, use_rgb=False, delay=0)
    for i, img in enumerate(crops):
        name = class_name[int(labels[i])] if class_name else labels[i]
        if out_dir:
            file_name = "{}_{:0=4d}_{}.jpg".format(image_id, i, flag) if flag else "{}_{:0=4d}.jpg".format(image_id, i)
            img_file = file_utils.create_dir(out_dir, name, file_name)
            cv2.imwrite(img_file, img)
        if vis: image_utils.cv_show_image("crop", img, use_rgb=False, delay=0)


if __name__ == "__main__":
    """
    对VOC的数据目标进行裁剪
    """
    # filename = "/home/PKing/nasdata/dataset/tmp/face_person/MPII/train5000.txt"
    filename = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-det-dataset/dataset-v4/file_list.txt"
    # class_name = ["face", "face-eyeglasses"]
    # class_name = "/home/dm/nasdata/dataset/tmp/traffic-sign/TT100K/VOC/train/class_name.txt"
    # class_name = ["unique"]
    class_name = ['身穿工作服', '未穿工作服', '表箱关', '表箱开', '其他鞋', '绝缘鞋']
    # class_name = ['person']
    out_dir = os.path.join(os.path.dirname(filename), "crops")
    dataset = parser_voc.VOCDataset(filename=filename,
                                    data_root=None,
                                    anno_dir=None,
                                    image_dir=None,
                                    class_name=class_name,
                                    transform=None,
                                    check=False,
                                    use_rgb=False,
                                    shuffle=False)
    print("have num:{}".format(len(dataset)))
    class_name = dataset.class_name
    scale = [1.0, 1.0]
    flag = str(scale[0]).replace(".", "p")
    flag = None
    # scale = None
    for i in tqdm(range(len(dataset))):
        try:
            data = dataset.__getitem__(i)
            image, targets, image_id = data["image"], data["target"], data["image_ids"]
            bboxes, labels = targets[:, 0:4], targets[:, 4:5]
            save_object_crops(image, out_dir, bboxes, labels, image_id, class_name=class_name, scale=scale,
                              flag=flag, vis=False)
        except Exception as e:
            print("error:{}".format(dataset.index2id(i)))
            print(e)
