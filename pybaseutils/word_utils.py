# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-03-29 16:15:09
    @Brief  :
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict
from pybaseutils import image_utils, json_utils, file_utils


def concat_stroke_image(mask, seg_list, split_line=False, vis=False):
    """
    水平拼接笔画图片
    :param mask: 整字的笔画mask
    :param seg_list: 字的笔画mask列表
    :param split_line: 是否显示分隔线
    :param vis: 是否可视化
    :return:返回水平拼接笔画图片
    """
    if len(seg_list) > 0:
        seg_mask = np.max(seg_list, axis=0)
    else:
        seg_mask = np.zeros_like(mask)
    diff = np.abs(mask - seg_mask)
    images = [mask] + [seg_mask] + [diff] + seg_list
    vis_image = image_utils.image_hstack(images, split_line=split_line)
    if vis:
        image_utils.cv_show_image("mask-seg-diff-stroke", vis_image, use_rgb=False)
    return vis_image


def concat_hw_gt_stroke_image(hw_mask, hw_segs, gt_mask, gt_segs, split_line=True, vis=False):
    """
    对比标准字的笔画和手写字的笔画分割图
    :param hw_mask:手写字整字mask
    :param hw_segs:手写字分割后的笔画mask列表
    :param gt_mask:标准字整字mask
    :param gt_segs:标准字真实的笔画mask列表
    :param split_line: 是否显示分隔线
    :param vis: 是否可视化
    :return:
    """
    hw_stroke = concat_stroke_image(hw_mask, hw_segs, split_line=split_line, vis=False)
    gt_stroke = concat_stroke_image(gt_mask, gt_segs, split_line=split_line, vis=False)
    vis_image = image_utils.image_vstack([gt_stroke, hw_stroke], split_line=split_line)
    if vis:
        image_utils.cv_show_image("gt-pd-stroke", vis_image, use_rgb=False)
    return vis_image


def show_word_info(word_info):
    print(json_utils.formatting(word_info['content'] if 'content' in word_info else ""))
    keys = ["label", "stroke_label", "stroke_names"]
    content = ["{}\t:{}".format(key, word_info[key]) for key in keys if key in word_info]
    print("\n".join(content))
    if len(word_info['stroke_segs']) == 0 or word_info['mask'] is None: return
    stroke_img = concat_stroke_image(word_info['mask'], word_info['stroke_segs'], split_line=True, vis=False)
    image_utils.cv_show_image("mask-seg-stroke", stroke_img, use_rgb=False)
    for i in range(len(word_info['piece_segs'])):
        stroke = word_info['stroke_segs'][i]
        piece = word_info['piece_segs'][i]
        names = word_info['stroke_label'][i]
        print("i={:0=3d}\tname={}\tpiece_num={}".format(i, names, len(piece)))
        if len(piece) == 0: continue
        piece_img = concat_stroke_image(stroke, piece, split_line=True, vis=False)
        image_utils.cv_show_image("mask-seg-piece", piece_img, use_rgb=False)
    cv2.destroyWindow("mask-seg-piece")
    print("---" * 20)


def show_hw_gt_word_info(word_info):
    """可视化标准字和手写字的分割效果"""
    keys = ["label", "stroke_label", "stroke_names"]
    for word in word_info:
        hw_word = word["handwriting"]
        gt_word = word["groundtruth"]
        print(json_utils.formatting(hw_word['content'] if 'content' in hw_word else ""))
        # print(json_utils.formatting(gt_info['content'] if 'content' in gt_info else ""))
        text = ["{}={}".format(key, hw_word[key]) for key in keys if key in hw_word]
        print("\t".join(text))
        if len(hw_word['stroke_segs']) == 0 or hw_word['mask'] is None: return
        concat_hw_gt_stroke_image(hw_mask=hw_word['mask'], hw_segs=hw_word['stroke_segs'],
                                  gt_mask=gt_word['mask'], gt_segs=gt_word['stroke_segs'], vis=True)
        for i in range(len(hw_word['piece_segs'])):
            hw_stroke = hw_word['stroke_segs'][i]
            hw_piece = hw_word['piece_segs'][i]
            gt_stroke = gt_word['stroke_segs'][i]
            gt_piece = gt_word['piece_segs'][i]
            names = hw_word['stroke_label'][i]
            print("i={:0=3d}\tname={}\tpiece_num={}".format(i, names, len(hw_piece)))
            if len(hw_piece) == 0: continue
            piece_image = concat_hw_gt_stroke_image(hw_mask=hw_stroke, hw_segs=hw_piece,
                                                    gt_mask=gt_stroke, gt_segs=gt_piece, vis=False)
            image_utils.cv_show_image("mask-seg-piece", piece_image)
        cv2.destroyWindow("mask-seg-piece")
        print("---" * 20)


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


def get_content_value(content: Dict, key: str):
    """在字典content中,查找包含key的所有数据"""
    r = None
    for k, v in content.items():
        if k == key:
            return v
        elif isinstance(v, Dict):
            r = get_content_value(v, key)
        elif isinstance(v, List):
            r = [get_content_value(c, key) for c in v if isinstance(c, Dict)]
    return r


def find_key_metadata(metadata: List[Dict], key):
    """在metadata中,查找包含key的所有数据"""
    values = []
    for content in metadata:
        v = get_content_value(content, key)
        values.append(v)
    return values


def show_word_packer(packer, image, keys=[], split_line=False, delay=0):
    _keys = ['label', 'cls_score', 'box', 'det_score']
    for word in packer:
        label = word['label'] if "label" in word else ""
        images = [word[k] for k in keys if k in word]
        info = ["{}:{}".format(k, word[k]) for k in _keys if k in word]
        print(info)
        if isinstance(image, np.ndarray):
            image = image_utils.draw_image_bboxes_text(image, boxes=[word["box"]],
                                                       boxes_name=[label], color=(0, 0, 255))
            show_images("dets", [image], split_line=split_line, delay=1)
        show_images("packer", images, split_line=split_line, delay=delay)


def show_word_unpacker(unpacker, image, keys=[], split_line=False, delay=0):
    _keys = ['label', 'cls_score', 'box', 'det_score']
    for i in range(len(unpacker["box"])):
        label = unpacker["label"][i] if "label" in unpacker else ""
        images = [unpacker[k][i] for k in keys if k in unpacker]
        info = ["{}:{}".format(k, unpacker[k][i]) for k in _keys if k in unpacker]
        print(info)
        if isinstance(image, np.ndarray):
            image = image_utils.draw_image_bboxes_text(image, boxes=[unpacker["box"][i]],
                                                       boxes_name=[label], color=(0, 0, 255))
            show_images("dets", [image], split_line=split_line, delay=1)
        show_images("unpacker", images, split_line=split_line, delay=delay)


def show_images(title, images, use_rgb=False, split_line=False, delay=0):
    if isinstance(images, np.ndarray): images = [images]
    if len(images) == 0: return
    image = image_utils.image_hstack(images, split_line=split_line)
    image_utils.cv_show_image(title, image, use_rgb=use_rgb, delay=delay)


def save_images(images, dsize=None, out_dir="./output"):
    for img in images:
        name = file_utils.get_time("p") + ".png"
        file = file_utils.create_dir(out_dir, None, name)
        img = image_utils.resize_image(img, size=dsize)
        cv2.imwrite(file, img)


def save_packer_images(packer, keys=["crop"], dsize=None, out_dir="./output"):
    for i, word in enumerate(packer):
        time = file_utils.get_time("p")
        for k in keys:
            name = "{:0=3d}_{}_{}.png".format(i, time, k)
            file = file_utils.create_dir(out_dir, None, name)
            img = image_utils.resize_image(word[k], size=dsize)
            cv2.imwrite(file, img)
