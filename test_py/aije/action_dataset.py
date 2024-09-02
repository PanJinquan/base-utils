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
from pybaseutils.dataloader import parser_labelme
from pybaseutils import image_utils, file_utils


def save_object_crops(data_info, out_dir, class_name=None, scale=[], square=False,
                      padding=False, min_size=20 * 20 * 3, flag="", vis=False):
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
    image, points, boxes, labels = data_info["image"], data_info["points"], data_info["boxes"], data_info["labels"]
    if len(boxes) == 0: return
    image_file = data_info["image_file"]
    h, w = image.shape[:2]
    image_id = os.path.basename(image_file).split(".")[0]
    if square:
        boxes = image_utils.get_square_boxes(boxes, use_max=True, baseline=-1)
    if scale:
        boxes = image_utils.extend_xyxy(boxes, scale=scale)
    if padding:
        crops = image_utils.get_bboxes_crop_padding(image, boxes)
    else:
        crops = image_utils.get_bboxes_crop(image, boxes)
    if vis:
        m = image_utils.draw_image_bboxes_labels(image.copy(), boxes, labels, class_name=class_name,
                                                 thickness=2, fontScale=0.8, drawType="chinese")
        image_utils.cv_show_image("image", m, use_rgb=False, delay=0)
    for i, img in enumerate(crops):
        if img.size < min_size: continue
        name = class_name[int(labels[i])] if class_name else labels[i]
        if out_dir:
            file_name = "{}_{:0=4d}_{}.jpg".format(image_id, i, flag) if flag else "{}_{:0=4d}.jpg".format(image_id, i)
            img_file = file_utils.create_dir(out_dir, name, file_name)
            cv2.imwrite(img_file, img)
        if vis: image_utils.cv_show_image("crop", img, use_rgb=False, delay=0)


if __name__ == "__main__":
    """
    对VOC的数据目标进行裁剪
    室内：['主杆', '从此进出标示牌', '其他鞋', '吊物绳', '在此工作标示牌', '垫子', '安全带', '安全帽', '安全绳',
          '导线头', '尖嘴钳', '工具袋', '手', '扳手', '抹布', '未穿工作服','柱式绝缘子', '止步高压危险标示牌',
          '绝缘手套', '绝缘鞋', '脚扣', '螺丝', '身穿工作服', '遮拦杆', '铁架', '铝扎线']
    室外: []
    """
    anno_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-action-det/aije-action-train-v01/json"
    class_name = None  # 室内
    out_dir = os.path.join(os.path.dirname(anno_dir), "crops")
    dataset = parser_labelme.LabelMeDatasets(filename=None,
                                             data_root=None,
                                             anno_dir=anno_dir,
                                             image_dir=None,
                                             class_name=class_name,
                                             check=False,
                                             phase="val",
                                             shuffle=False)
    print("have num:{}".format(len(dataset)))
    class_name = dataset.class_name
    scale = [2.0, 2.0]
    flag = str(scale[0]).replace(".", "p")
    flag = None
    # scale = None
    for i in tqdm(range(len(dataset))):
        data_info = dataset.__getitem__(i)
        save_object_crops(data_info, out_dir, class_name=class_name, scale=scale, flag=flag, vis=False)
