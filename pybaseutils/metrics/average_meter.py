# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-10-08 08:56:25
    @Brief  :
"""
import numpy as np


class AverageMeter(object):
    """计算并存储参数当前值或平均值
    Computes and stores the average and current value
    -------------------------------------------------
    batch_time = AverageMeter()
    即 self = batch_time
    则 batch_time 具有__init__，reset，update三个属性，
    直接使用batch_time.update()调用
    功能为：batch_time.update(time.time() - end)
       仅一个参数，则直接保存参数值
    对应定义：def update(self, val, n=1)
    losses.update(loss.data[0], input.size(0))
    top1.update(prec1[0], input.size(0))
    top5.update(prec5[0], input.size(0))
    这些有两个参数则求参数val的均值，保存在avg中
    """

    def __init__(self):
        self.reset()  # __init__():reset parameters

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if n == 0:
            return
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MultiAverageMeter(object):
    def __init__(self, keys: list):
        self.keys = keys
        self.avg_meters = {}
        for k in keys: self.avg_meters[k] = AverageMeter()

    def reset(self, keys=None):
        if isinstance(keys, str):
            self.avg_meters[keys].reset()
        elif isinstance(keys, list):
            for k in keys: self.avg_meters[k].reset()
        else:
            for k in self.avg_meters.keys(): self.avg_meters[k].reset()

    def update(self, items: dict, n=1):
        for k, v in items.items(): self.avg_meters[k].update(v, n)


def accuracy(output, target, topk=(1, 5)):
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
