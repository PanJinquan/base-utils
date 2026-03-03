# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : 2026-02-02 09:28:16
# @Brief  :
# --------------------------------------------------------
"""
import os
import cv2
import numpy as np
import random
from tqdm import tqdm
from pybaseutils import image_utils, file_utils


def motion_blur(image, degree=12, angle=0):
    image = np.array(image)
    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    kernel = np.diag(np.ones(degree))
    kernel = cv2.warpAffine(kernel, M, (degree, degree))
    kernel = kernel / degree
    blurred = cv2.filter2D(image, -1, kernel)
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred


def random_motion_blur(image, degrees=[25, 35], angle=[0, 180]):
    degree = random.uniform(degrees[0], degrees[1])
    angle = random.uniform(angle[0], angle[1])
    blurred = motion_blur(image, degree=int(degree), angle=angle)
    return blurred, degree, angle


def task_motion_blur(image_root, class_maps, image_outs, class_dict={}, vis=False):
    file_utils.create_dir(image_outs)
    file = file_utils.write_file(os.path.join(image_outs, '分类数据集，请勿删除.txt'), data='', mode='w')
    for name in class_maps:
        image_dir = f"{image_root}/{name}"
        sub = class_dict[name] if class_dict else name
        print(image_dir)
        image_list = file_utils.get_images_list(image_dir)
        if len(image_list) == 0: continue
        out_dir = os.path.join(image_outs, sub)
        os.makedirs(out_dir, exist_ok=True)
        for file in tqdm(image_list):
            try:
                out_file = os.path.join(out_dir, os.path.basename(file))
                img = cv2.imread(file)
                # img = image_utils.resize_image(img, (None, 224))
                img = image_utils.resize_image_padding(img, (224, 224))
                dst, degree, angle = random_motion_blur(img)
                cv2.imwrite(out_file, dst)
            except Exception as e:
                print(f"error:{file}")
            if vis:
                fus = image_utils.image_hstack([img, dst])
                image_utils.cv_show_image("image:{}".format(img.shape), fus)
                cv2.destroyAllWindows()


if __name__ == "__main__":
    class_maps = {'橙色Pe标签': '橙色模糊标签',
                  '紫色Qr标签': '紫色模糊标签',
                  '红色A0标签': '红色模糊标签', '红色B1标签': '红色模糊标签', '红色C2标签': '红色模糊标签',
                  '红色D3标签': '红色模糊标签', '红色H5标签': '红色模糊标签',
                  '绿色Su标签': '绿色模糊标签', '绿色Tv标签': '绿色模糊标签',
                  '蓝色K6标签': '蓝色模糊标签', '蓝色L7标签': '蓝色模糊标签', '蓝色Na标签': '蓝色模糊标签',
                  '青色E4标签': '青色模糊标签',
                  '黄色Rh标签': '黄色模糊标签'}

    # class_maps = {'红色A0标签': '红色模糊标签', '红色B1标签': '红色模糊标签', '红色C2标签': '红色模糊标签',
    #               '红色D3标签': '红色模糊标签', '红色H5标签': '红色模糊标签'}

    # 示例用法
    datasets = [
        '/home/PKing/nasdata/dataset-dmai/AILT/ailt-det/dataset-20260120/crops',
        '/home/PKing/nasdata/dataset-dmai/AILT/ailt-det/dataset-20260130/crops',
        '/home/PKing/nasdata/dataset-dmai/AILT/ailt-det/dataset-20260206/crops',
    ]

    for image_root in datasets:
        print(image_root)
        assert os.path.exists(image_root), image_root
        out_dir = os.path.join(os.path.dirname(image_root), "blurs")
        if os.path.exists(out_dir) and out_dir.endswith('blurs'): file_utils.remove_dir(out_dir)
        task_motion_blur(image_root, class_maps, image_outs=out_dir, class_dict=class_maps, vis=False)
