# -*- coding: utf-8 -*-
'''
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-11-29 18:11:47
    @Brief  :
'''
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
    '''
    对labelme的数据目标进行裁剪，用于制作分类数据集
    '''

    target_name = ['身穿工作服', '未穿工作服',
                   '绝缘鞋', '脚穿绝缘鞋', '长筒靴', '脚穿长筒靴', '其他鞋', '脚穿其他鞋',
                   '手', '绝缘手套', '手穿绝缘手套', '棉纱手套', '手穿棉纱手套', '其他手套', '手穿其他手套',
                   '拨动开关分闸', '拨动开关合闸',
                   '电容柜开关合闸', '电容柜开关分闸',
                   '低压侧总刀闸合闸', '低压侧总刀闸分闸',
                   '面板单键开关断开', '面板单键开关合上',
                   '面板双键开关断开', '面板双键开关合上',
                   '柱上开关刀闸分闸', '柱上开关刀闸合闸',
                   '低压开关断开', '低压开关合上',
                   '表箱关', '表箱开', '其他',
                   ]
    # 还差核相的数据
    datasets = [
        # TODO 训练集
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/01-核相操作/dataset-v01/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/01-核相操作/dataset-v02/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/01-核相操作/dataset-v03/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/01-核相操作/dataset-v04/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/02-柱上开关/dataset-v01/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/03-架空线接地/dataset-v01/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/05-装设遮拦/dataset-v01/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/06-地网电阻/dataset-v01/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/06-地网电阻/dataset-v02/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/08-更换熔丝/dataset-v01/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/08-更换熔丝/dataset-v02/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/08-更换熔丝/dataset-v20/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/08-更换熔丝/dataset-v21/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/08-更换熔丝/dataset-v23-special/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/08-更换熔丝/dataset-v24/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/09-更换熔断器/dataset-v20/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/09-更换熔断器/dataset-v21/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/09-更换熔断器/dataset-v22/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/10-钳形电流表/dataset-v01/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/10-钳形电流表/dataset-v02-special/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/10-钳形电流表/dataset-v03/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/10-钳形电流表/dataset-v04-special/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/13-工器具演示/dataset-v21/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/13-工器具演示/dataset-v22/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/22-柱开巡视/dataset-v01/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/23-更换绝缘子/dataset-v01/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/23-更换绝缘子/dataset-v02/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/23-更换绝缘子/dataset-v03/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/23-更换绝缘子/dataset-v04/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/23-更换绝缘子/dataset-v05/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/23-更换绝缘子/dataset-v06/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/23-更换绝缘子/dataset-v07/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/25-拉线检查/dataset-v01/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/31-临时遮拦/dataset-v01/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/31-临时遮拦/dataset-v02/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/31-临时遮拦/dataset-v03/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/31-临时遮拦/dataset-v04/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/32-绝缘包扎/dataset-v01/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/32-绝缘包扎/dataset-v02/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/33-日常巡视/dataset-v01/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/33-日常巡视/dataset-v02/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/35-清除飘挂物/dataset-v01/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/35-清除飘挂物/dataset-v21/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/35-清除飘挂物/dataset-v22/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/35-清除飘挂物/dataset-v23/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/37-带电接火/dataset-v01/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/36-低压开关/dataset-v01/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/36-低压开关/dataset-v02/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/37-带电接火/dataset-v01/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/37-带电接火/dataset-v02/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/37-带电接火/dataset-v03/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/37-带电接火/dataset-v04-special/images',
        # TODO 测试集
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/01-核相操作/dataset-test-v01/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/01-核相操作/dataset-v03-val/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/01-核相操作/dataset-v04-val/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/02-柱上开关/dataset-v01-val/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/03-架空线接地/dataset-v01-val/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/05-装设遮拦/dataset-v01-val/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/06-地网电阻/dataset-v01-val/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/06-地网电阻/dataset-v02-val/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/08-更换熔丝/dataset-v21-val/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/08-更换熔丝/dataset-v23-special-val/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/08-更换熔丝/dataset-v24-val/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/09-更换熔断器/dataset-v21-val/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/09-更换熔断器/dataset-v22-val/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/10-钳形电流表/dataset-v01-val/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/10-钳形电流表/dataset-v02-special-val/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/10-钳形电流表/dataset-v03-val/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/10-钳形电流表/dataset-v04-special-val/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/13-工器具演示/dataset-v22-val/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/22-柱开巡视/dataset-v01-val/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/23-更换绝缘子/dataset-test-v01/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/25-拉线检查/dataset-v01-val/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/32-绝缘包扎/dataset-v01-val/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/32-绝缘包扎/dataset-v02-val/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/33-日常巡视/dataset-v01-val/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/33-日常巡视/dataset-v02-val/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/35-清除飘挂物/dataset-v22-val/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/35-清除飘挂物/dataset-v23-val/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/36-低压开关/dataset-v01-val/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/36-低压开关/dataset-v02-val/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/37-带电接火/dataset-v02-val/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/37-带电接火/dataset-v03-val/images',
        '/home/PKing/nasdata/dataset-dmai/AIJE/dataset/aije-v2-det/37-带电接火/dataset-v04-special-val/images',
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
