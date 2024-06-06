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
from pybaseutils import image_utils, file_utils, coords_utils
from pybaseutils.transforms import transform_utils


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
    image, points, bboxes, labels = data_info["image"], data_info["points"], data_info["boxes"], data_info["labels"]
    if len(bboxes) == 0: return
    image_file = data_info["image_file"]
    h, w = image.shape[:2]
    image_id = os.path.basename(image_file).split(".")[0]
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


def save_object_crops_aligment(data_info, out_dir, class_name=None, scale=(1.0, 1.0), vis=False, delay=0):
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
    flag = str(scale[0]).replace(".", "p")
    file_utils.create_dir(out_dir)
    image, points, bboxes, labels = data_info["image"], data_info["points"], data_info["boxes"], data_info["labels"]
    if len(bboxes) == 0: return
    image_file = data_info["image_file"]
    h, w = image.shape[:2]
    image_id = os.path.basename(image_file).split(".")[0]
    for i in range(len(points)):
        point, label, box = points[i], labels[i], bboxes[i]
        src_pts = transform_utils.get_obb_points(point)
        crop, dst_pts, M, Minv = transform_utils.image_alignment(image, src_pts, dst_pts=None, scale=scale)
        h, w = crop.shape[:2]
        name = "{}_{}".format(image_id, flag) if flag else image_id
        if w > 2.5 * h:
            crop_file = os.path.join(out_dir, "labels", "{}_{}_v1.jpg".format(label, name))
        else:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
            crop_file = os.path.join(out_dir, "others", "{}_{}_v1.jpg".format(label, name))
        file_utils.create_file_path(crop_file)
        cv2.imwrite(crop_file, crop)
        if vis:
            image = image_utils.draw_image_bboxes_text(image, [box], boxes_name=[label])
            image = image_utils.draw_key_point_in_image(image, [src_pts], vis_id=True)
            image_utils.cv_show_image("crop", crop, delay=10)
            image_utils.cv_show_image("image", image, delay=delay)


if __name__ == "__main__":
    """"""
    anno_dir = "/home/PKing/nasdata/tmp/tmp/WaterMeter/水表数据集/Water-Meter-Det3/train/data-frame/json"
    class_name = None
    out_dir = os.path.join(os.path.dirname(anno_dir), "crops")
    dataset = parser_labelme.LabelMeDataset(filename=None,
                                            data_root=None,
                                            anno_dir=anno_dir,
                                            image_dir=None,
                                            class_name=class_name,
                                            check=False,
                                            phase="val",
                                            shuffle=False)
    print("have num:{}".format(len(dataset)))
    rati = 1.0
    class_name = dataset.class_name
    scale = [1.00, 1.0]

    # scale = [1.00, 1.00]
    # flag = None
    # /home/PKing/nasdata/tmp/tmp/WaterMeter/水表数据集/Water-Meter-Det2/train/images/id_1161_value_71_024.jpg
    for i in tqdm(range(len(dataset))):
        data_info = dataset.__getitem__(i)
        # save_object_crops(data_info, out_dir, class_name=class_name, scale=scale, flag=flag, vis=False)
        save_object_crops_aligment(data_info, out_dir, scale=scale, class_name=class_name, vis=False)
