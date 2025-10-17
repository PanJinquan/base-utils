# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : 2025-08-20 16:04:27
# @Brief  :
# --------------------------------------------------------
"""
import json
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict


class Objects(object):
    def __init__(self, boxes, label, score, names=[], size=[], segms=None, masks=None, keypt=None,
                 traid=None, trace=None, color=[], image=None):
        self.boxes = boxes  # 目标框(xmin,ymin,xmax,ymax)
        self.score = score  # 目标框置信度
        self.label = label  # 目标框类别
        self.keypt = keypt  # 目标关键点
        self.names = names  # 类别名称
        self.image = image  # 原始图像
        self.masks = masks  # 目标轮廓掩码
        self.segms = segms  # 目标轮廓[np.shape(n,2)]
        self.traid = traid  # 轨迹ID
        self.trace = trace  # 轨迹的历史boxes
        self.color = [(255, 0, 0)] * len(boxes)  # 目标框颜色
        self.size = size  # 原始图像大小
        self.depth = None  # 深度图
        self.metas = defaultdict()  # 其他信息

    def update(self, **kwargs):
        """更新属性值"""
        for k, v in kwargs.items():
            setattr(self, k, v) if hasattr(self, k) else None

    def todict(self, keys: List = []):
        """将属性转为字典"""
        out = self.__dict__
        if keys: out = {k: out.get(k, None) for k in keys}
        return out

    def __str__(self):
        keys = ["names", "boxes", "score", "label", "segms"]
        msgs = ""
        for k, v in self.todict(keys=keys).items():
            msgs += "{}:{}\n".format(k, v)
        return msgs

    def get_obj(self, names: List, keys: List = []):
        """选择目标,返回新的Objects对象"""
        obj = Objects(boxes=[], label=[], score=[])
        names = dict.fromkeys(names, 0)
        index = [i for i, n in enumerate(self.names) if n in names]
        for k, v in self.__dict__.items():
            if k in keys:
                v = v[index] if isinstance(v, np.ndarray) else [v[i] for i in index]
            setattr(obj, k, v)
        return obj

    def del_obj(self, names: List, keys: List = []):
        """删除目标,返回新的Objects对象"""
        obj = Objects(boxes=[], label=[], score=[])
        names = dict.fromkeys(names, 0)
        index = [i for i, n in enumerate(self.names) if n not in names]
        for k, v in self.__dict__.items():
            if k in keys:
                v = v[index] if isinstance(v, np.ndarray) else [v[i] for i in index]
            setattr(obj, k, v)
        return obj


class Dict2Obj(object):
    """ dict转类对象"""

    def __init__(self, args):
        self.__dict__.update(args)


def str2dict(data: str):
    try:
        return json.loads(data)
    except Exception as e:
        print(e)
    return None


def dict_slice(data: Dict, slice=[]):
    """
    字典像列表一样切片
    :param data:
    :param slice:范围，如(2,3]表示选择第2到第3项(不含)(索引从0开始)
    :return:
    """
    out = dict(list(data.items())[slice[0]:slice[1]])
    return out


def dict_sort(data: Dict, reverse=False, use_key=True):
    """
    按照字典的key/value值排序
    :param data:
    :param reverse: False 升序,True  降序
    :param use_key: True使用key进行排序，False使用value进行排序
    """
    if use_key:
        dst = dict(sorted(data.items(), key=lambda x: x[0], reverse=reverse))
    else:
        dst = dict(sorted(data.items(), key=lambda x: x[1], reverse=reverse))
    return dst


if __name__ == "__main__":
    boxes = np.array([[100, 100, 200, 200], [300, 300, 400, 400]])
    score = np.array([0.9, 0.8])
    label = np.array([0, 1])
    names = ["cat", "dog"]
    obj1 = Objects(boxes, label, score, names, segms="segms111", masks=boxes)
    obj2 = obj1.get_obj(["dog", "hao"], ["names", "boxes", "score", "label"])
    print(obj1)
    print(obj2)
