# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2024-10-09 14:19:18
    @Brief  :
"""
import cv2
import numpy as np
import torch
import torchvision
from pybaseutils import image_utils


def nms_boxes_cv2(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, score_threshold=0.5, nms_threshold=0.5,
                  top_k=None, use_batch=True):
    """
    :param boxes:
    :param scores:
    :param labels:
    :param score_threshold:
    :param nms_threshold:
    :param top_k:
    :return:
    """
    if use_batch:
        # TODO opencv4.7.0版本中增加了NMSBoxesBatched函数，可分类做nms
        xywh = image_utils.xyxy2xywh(boxes)
        index = cv2.dnn.NMSBoxesBatched(xywh.tolist(), scores.tolist(), labels.tolist(),
                                        score_threshold=score_threshold, nms_threshold=nms_threshold, top_k=top_k)
    else:
        # TODO CV2 NMSBoxes没有区分类别
        # 如果你不想每个类别都做一次nms,而是所有类别一起做nms
        # 就需要把不同类别的目标框尽量没有重合，不至于把不同类别的IOU大的目标框滤掉
        # 先用每个类别id乘一个很大的数，作为offset,把每个类别的box坐标都加上相应的offset,这是batched nms
        max_wh = 7680
        c = (labels * max_wh).reshape(-1, 1).astype(np.float32)  # classes
        boxes_ = boxes + c  # boxes (offset by class)
        xywh = image_utils.xyxy2xywh(boxes_)
        index = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), score_threshold=score_threshold,
                                 nms_threshold=nms_threshold,top_k=top_k)
    return index


def nms_boxes_torch(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, score_threshold=0.5, nms_threshold=0.5,
                    top_k=None):
    """
    :param boxes:
    :param scores:
    :param labels:
    :param score_threshold:
    :param nms_threshold:
    :param top_k:
    :return:
    """
    max_wh = 7680
    c = (labels * max_wh).reshape(-1, 1).astype(np.float32)  # classes
    boxes_ = boxes + c  # boxes (offset by class)
    index = torchvision.ops.nms(torch.from_numpy(boxes_),
                                torch.from_numpy(scores),
                                iou_threshold=nms_threshold)  # NMS
    return index
