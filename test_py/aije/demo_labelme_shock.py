# -*- coding: utf-8 -*-
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
from pybaseutils import image_utils, file_utils, json_utils


def save_object_crops(data_info, item_list, out_dir, class_name=None, scale=[], square=False,
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
    image_file, image = data_info["image_file"], data_info["image"]
    image_id = os.path.basename(image_file).split(".")[0]
    h, w = image.shape[:2]
    for i, item in enumerate(item_list):
        if len(item['box']) == 0: continue
        bboxes = [item['box']]
        labels = [item['name']]
        attrib_name = json_utils.get_value(item, key=['attribute', 'names'], default=[])
        if square:
            bboxes = image_utils.get_square_boxes(bboxes, use_max=True, baseline=-1)
        if scale:
            bboxes = image_utils.extend_xyxy(bboxes, scale=scale)
        if padding:
            crops = image_utils.get_bboxes_crop_padding(image, bboxes)
        else:
            crops = image_utils.get_bboxes_crop(image, bboxes)
        img = crops[0]
        if img.size < min_size: continue
        name = "_".join(labels + list(set(attrib_name)))
        if out_dir:
            file_name = "{}_{:0=4d}_{}.jpg".format(image_id, i, flag) if flag else "{}_{:0=4d}.jpg".format(image_id,
                                                                                                           i)
            img_file = file_utils.create_dir(out_dir, name, file_name)
            cv2.imwrite(img_file, img)
        if vis:
            m = image_utils.draw_image_bboxes_labels(image.copy(), bboxes, labels, class_name=class_name,
                                                     thickness=2, fontScale=0.8, drawType="chinese")
            image_utils.cv_show_image("image", m, use_rgb=False, delay=0)


if __name__ == "__main__":
    """
    对VOC的数据目标进行裁剪
    室内：['主杆', '从此进出标示牌', '其他鞋', '吊物绳', '在此工作标示牌', '垫子', '安全带', '安全帽', '安全绳',
          '导线头', '尖嘴钳', '工具袋', '手', '扳手', '抹布', '未穿工作服','柱式绝缘子', '止步高压危险标示牌',
          '绝缘手套', '绝缘鞋', '脚扣', '螺丝', '身穿工作服', '遮拦杆', '铁架', '铝扎线']
    室外: []
    """
    # TODO 其他试题
    anno_dir = [
        "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/dataset-v21/images",
        "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/dataset-v22/images",
        "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/dataset-v23/images",
        "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/dataset-v24/images",
        "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/dataset-v26/images",
        "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/dataset-v27/images",
        "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/dataset-v29/images",
        "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/dataset-v29-test/images",
    ]
    out_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-shock/dataset-others/待分类"

    # TODO 更换绝缘子
    # anno_dir = [
        # "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/dataset-v25/images",
        # "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/dataset-v28/images",
        # "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-shock/dataset-sample/images",
    # ]
    # out_dir = "/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-shock/dataset-v1/待分类"

    superclass = ["主杆"]
    subclass = ['未穿工作服', '身穿工作服']
    class_name = superclass + subclass
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
    scale = [1.0, 1.0]
    flag = str(scale[0]).replace(".", "p")
    flag = None
    for i in tqdm(range(len(dataset))):
        data_info = dataset.__getitem__(i)
        item_list = parser_labelme.LabelMeDataset.get_sub2superclass(data_info,
                                                                     superclass=superclass,
                                                                     subclass=subclass,
                                                                     square=True,
                                                                     scale=scale,
                                                                     vis=False)
        save_object_crops(data_info, item_list, out_dir, flag=flag, vis=False)
