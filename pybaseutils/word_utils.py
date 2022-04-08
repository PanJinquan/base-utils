# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-03-29 16:15:09
    @Brief  :
"""
import numpy as np
from typing import List, Tuple, Dict
from pybaseutils import image_utils


def concat_stroke_image(mask, seg_list, vis=False):
    """
    水平拼接笔画图片
    :param mask: 整字的笔画mask
    :param seg_list: 字的笔画mask列表
    :param vis: 是否可视化
    :return:返回水平拼接笔画图片
    """
    seg_mask = np.max(seg_list, axis=0)
    images = [mask] + [seg_mask] + seg_list
    vis_image = np.concatenate(images, axis=-1)
    if vis:
        image_utils.cv_show_image("mask-seg-stroke", vis_image, use_rgb=False)
    return vis_image


def concat_pd_gt_stroke_image(pd_mask, pd_segs, gt_mask, gt_segs, vis=False):
    """
    对比标准字的笔画和待测字的笔画分割图
    :param pd_mask:待测字整字mask
    :param pd_segs:待测字分割后的笔画mask列表
    :param gt_mask:标准字整字mask
    :param gt_segs:标准字真实的笔画mask列表
    :param vis: 是否可视化
    :return:
    """
    pd_stroke = concat_stroke_image(pd_mask, pd_segs, vis=False)
    gt_stroke = concat_stroke_image(gt_mask, gt_segs, vis=False)
    vis_image = image_utils.image_vstack([gt_stroke, pd_stroke])
    if vis:
        image_utils.cv_show_image("gt-pd-stroke", vis_image, use_rgb=False)
    return vis_image


def concat_packer(packers: List[List[Dict]]):
    if len(packers) == 0:
        return []
    if len(packers) == 1:
        return packers
    out_packers = packers[0]
    nums = len(packers[0])
    for p in range(1, len(packers)):
        assert nums == len(packers[p])
        for i in range(nums):
            out_packers[i].update(packers[p][i])
    return out_packers


def word_packer(unpacker: Dict, keys):
    packer = []
    for key in keys:
        if not packer:
            packer = [{key: v} for v in unpacker[key]]
        else:
            for i in range(len(unpacker[key])):
                packer[i][key] = unpacker[key][i]
    return packer


def word_unpacker(packer: List[Dict], keys):
    unpacker = {key: [] for key in keys}
    for i in range(len(packer)):
        for key, value in packer[i].items():
            if key in keys:
                unpacker[key].append(value)
    return unpacker


def show_word_packer(packer, image, keys=[], delay=0):
    _keys = ['label', 'cls_score', 'box', 'det_score']
    for word in packer:
        label = word['label'] if "label" in word else ""
        image = image_utils.draw_image_bboxes_text(image, boxes=[word["box"]], boxes_name=[label], color=(0, 0, 255))
        images = [word[k] for k in keys if k in word]
        info = ["{}:{}".format(k, word[k]) for k in _keys if k in word]
        print(info)
        show_image("dets", [image], delay=1)
        show_image("packer", images, delay=delay)


def show_word_unpacker(unpacker, image, keys=[], waitKey=0):
    _keys = ['label', 'cls_score', 'box', 'det_score']
    for i in range(len(unpacker["box"])):
        label = unpacker["label"][i] if "label" in unpacker else ""
        image = image_utils.draw_image_bboxes_text(image, boxes=[unpacker["box"][i]],
                                                   boxes_name=[label], color=(0, 0, 255))
        images = [unpacker[k][i] for k in keys if k in unpacker]
        info = ["{}:{}".format(k, unpacker[k][i]) for k in _keys if k in unpacker]
        print(info)
        show_image("dets", [image], delay=1)
        show_image("unpacker", images, delay=waitKey)


def show_image(title, images, use_rgb=False, delay=0):
    if isinstance(images, np.ndarray): images = [images]
    if len(images) == 0: return
    image = image_utils.image_hstack(images)
    image_utils.cv_show_image(title, image, use_rgb=use_rgb, delay=delay)
