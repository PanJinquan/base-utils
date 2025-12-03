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
from pybaseutils import image_utils, file_utils


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
                out_dir_ = out_dir if name in target_name else os.path.join(out_dir, '其他')
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
    # 接地引线'
    names = ['保护屏标示牌', '显示屏', '通道切换开关', '信号复归按钮', '红色圆形标签,黄色圆形标签', '地极端子',
             '地极标签',
             '红色压板开关合上,红色压板开关断开,白色压板开关合上,白色压板开关断开', '重合闸把手停用,重合闸把手单重',
             '本体端子箱关闭,本体端子箱打开', '主变端子箱关闭,主变端子箱打开', '分接开关箱关闭,分接开关箱打开',
             '主变压器标示牌', '端子箱标示牌', '油温表', '绕组温度表', '油位指示标示牌', '本体阀门', '呼吸器油杯',
             '沉降观察点标示牌', '冷却器控制箱打开,冷却器控制箱关闭', '冷却器', '风扇', '潜油泵', '油流指示器', '喷淋启动装置'
             ]
    target_name = ['身穿工作服', '未穿工作服',
                   '绝缘鞋', '脚穿绝缘鞋', '长筒靴', '脚穿长筒靴', '其他鞋', '脚穿其他鞋',
                   '手', '绝缘手套', '手穿绝缘手套', '棉纱手套', '手穿棉纱手套', '其他手套', '手穿其他手套',
                   '拨动开关分闸', '拨动开关合闸',  # 08-更换熔丝
                   '电容柜开关合闸', '电容柜开关分闸',  # 08-更换熔丝
                   '低压侧总刀闸合闸', '低压侧总刀闸分闸',  # 08-更换熔丝
                   '面板单键开关断开', '面板单键开关合上',  # 22-柱开巡视
                   '面板双键开关断开', '面板双键开关合上',  # 22-柱开巡视
                   '柱上开关刀闸分闸', '柱上开关刀闸合闸',  # 01-核相操作 02-柱上开关
                   '低压开关断开', '低压开关合上',  # 32-绝缘包扎 36-低压开关，37-带电接火
                   '表箱关', '表箱开', '其他',
                   "红色圆形标签", "黄色圆形标签",
                   "白色压板开关合上", "白色压板开关断开", "红色压板开关合上", "红色压板开关断开",
                   "重合闸把手停用", "重合闸把手单重",
                   '本体端子箱关闭', '本体端子箱打开',
                   '主变端子箱关闭', '主变端子箱打开',
                   '分接开关箱关闭', '分接开关箱打开',
                   '冷却器控制箱打开', '冷却器控制箱关闭',
                   '万用表线头', '万用表红色线头', '万用表黑色线头'
                   ]
    # 还差核相的数据
    datasets = [
        # TODO 训练集
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/02-东莞-220KV主变压器日常巡视/dataset-test01/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/02-东莞-220KV主变压器日常巡视/dataset-test02/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/01-东莞-投退重合闸操作/dataset-test01/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/01-东莞-投退重合闸操作/dataset-test02/images',

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
