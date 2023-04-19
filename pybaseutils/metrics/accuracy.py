# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-11-14 11:35:42
    @Brief  :
"""
import torch
import numpy as np
from sklearn import metrics


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    """
    计算topK的准确率，numpy_utils.py的get_topK
    """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k / batch_size)
    return res


def accuracy_score(true_labels, pred_labels):
    return metrics.accuracy_score(y_true=true_labels, y_pred=pred_labels)
