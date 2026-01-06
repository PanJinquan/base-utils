# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2025-07-22 10:40:45
# @Brief  :
# --------------------------------------------------------
"""
import os
import cv2
from tqdm import tqdm
from pybaseutils.dataloader import parser_labelme
from pybaseutils import image_utils, file_utils, text_utils


def save_object_crops(data_info, out_dir, class_name=None, target_name=None, scale=[], square=False,
                      padding=False, min_size=20 * 20 * 3, flag='', vis=False):
    '''
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
    '''
    image, points, bboxes, labels = data_info['image'], data_info['points'], data_info['boxes'], data_info['labels']
    if len(bboxes) == 0: return
    image_file = data_info['image_file']
    h, w = image.shape[:2]
    image_id, img_postfix = file_utils.split_postfix(image_file)
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
                                                 thickness=2, fontScale=0.8, drawType='chinese')
        image_utils.cv_show_image('image', m, use_rgb=False, delay=0)
    for i, img in enumerate(crops):
        if img.size < min_size: continue
        name = class_name[int(labels[i])] if class_name else labels[i]
        if out_dir:
            file_name = '{}_{:0=4d}_{}.jpg'.format(image_id, i, flag) if flag else '{}_{:0=4d}.jpg'.format(image_id, i)
            if target_name:
                # out_dir_ = out_dir if name in target_name else os.path.join(out_dir, '其他')
                m = text_utils.find_match_texts([name], target_name)
                out_dir_ = out_dir if m else os.path.join(out_dir, '其他')
                img_file = file_utils.create_dir(out_dir_, name, file_name)
            else:
                img_file = file_utils.create_dir(out_dir, name, file_name)
            cv2.imwrite(img_file, img)
        if vis: image_utils.cv_show_image('crop', img, use_rgb=False, delay=0)


def crop_dataset(anno_dir, out_dir=None, class_name=None, target_name=None):
    dataset = parser_labelme.LabelMeDatasets(filename=None,
                                             data_root=None,
                                             anno_dir=anno_dir,
                                             image_dir=None,
                                             class_name=class_name,
                                             check=False,
                                             phase='val',
                                             shuffle=False)
    print('have num:{}'.format(len(dataset)))
    class_name = dataset.class_name
    scale = [1.0, 1.0]
    # flag = str(scale[0]).replace('.', 'p')
    flag = None
    # scale = None
    for i in tqdm(range(len(dataset))):
        data_info = dataset.__getitem__(i)
        save_object_crops(data_info, out_dir, class_name=class_name, target_name=target_name, scale=scale, flag=flag,
                          vis=False)


def get_sub_dataset(data_root):
    sub_paths = file_utils.get_sub_paths(data_root, abspath=True)
    datasets = []
    for sub_path in sub_paths:
        data_list = file_utils.get_sub_paths(sub_path, abspath=True)
        data_list = [os.path.join(p, 'images') for p in data_list if os.path.exists(os.path.join(p, 'images'))]
        datasets += data_list
    return datasets


if __name__ == '__main__':
    """
    对labelme的数据目标进行裁剪，用于制作分类数据集
    """
    # TODO 蓝色长条标签和鞋套是否相似
    target_name = ['身穿工作服', '未穿工作服',
                   '绝缘鞋', '脚穿绝缘鞋', '长筒靴', '脚穿长筒靴', '其他鞋', '脚穿其他鞋', '脚穿鞋套',
                   '手', '绝缘手套', '手穿绝缘手套', '棉纱手套', '手穿棉纱手套', '其他手套', '手穿其他手套',
                   '胶枪灯', '胶枪灯红色', '胶枪灯绿色',
                   'PDA', 'PDA*', '手与*接触', '电视柜台面'
                   ]
    # '*长条标签'
    datasets = [
        '/home/PKing/nasdata/dataset-dmai/AILT/ailt-det/dataset-20251208-PDA/images',
        # '/home/PKing/nasdata/dataset-dmai/AILT/ailt-det/dataset-20251208-PDA-val/images',
    ]
    # dataroot = '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det'
    # datasets = get_sub_dataset(dataroot)
    class_name = []
    datasets = sorted(datasets)
    for anno_dir in datasets:
        print(anno_dir)
        assert os.path.exists(anno_dir), anno_dir
        out_dir = os.path.join(os.path.dirname(anno_dir), 'crops')
        if os.path.exists(out_dir) and out_dir.endswith('crops'): file_utils.remove_dir(out_dir)
        file_utils.create_dir(out_dir)
        file = file_utils.write_file(os.path.join(out_dir, '分类数据集，请勿删除.txt'), data='', mode='w')
        crop_dataset(anno_dir, out_dir=out_dir, class_name=class_name, target_name=target_name)
