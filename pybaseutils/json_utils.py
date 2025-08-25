# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2022-04-29 09:13:09
# @Brief  :
# --------------------------------------------------------
"""
import os
import toolz
import json
import numbers
from collections import Counter
from typing import List, Tuple, Dict
from pybaseutils.file_utils import load_json, read_json_data, save_json, write_json_path
from pybaseutils.dict_uils import *


def get_most_common(data: list, topK=None):
    """
    Counter类来统计列表元素的出现次数，然后找到出现次数最多的元素
    :param data:
    :return:
    """
    counter = Counter(data)  # # 使用 Counter 统计元素出现的次数
    # 返回TopK，其中包含出现次数最多的元素及其次数（按从高到低排序)
    return counter.most_common(topK)


def formatting(data):
    """格式化json数据"""
    info = json.dumps(data, indent=1, separators=(', ', ': '), ensure_ascii=False)
    return info


def get_keys_vaules(data, func=None):
    """
    遍历json数据并获得所有value的key路径
    :param data:
    :param func: 过滤条件函数func(k,v),默认为None,表示获取有的,获得所有value的key路径,一些常用的过滤方法：
           过滤所有文件：func = lambda k,v: isinstance(v, str) and os.path.isfile(v) and os.path.exists(v)
           过滤所有字符串：func = lambda k,v: isinstance(v, str)
           过滤所有数字：func = lambda k,v: isinstance(v, numbers.Number)
    :return: 返回满足条件的keys, values
    """

    def recursion(value, key=None, sub=[]):
        if not key is None: sub.append(key)
        if isinstance(value, list):
            for i in range(len(value)):
                recursion(value[i], key=i)
        elif isinstance(value, dict):
            for k, v in value.items():
                recursion(v, key=k)
        elif func is None:
            keys.append(sub.copy())
            values.append(value)
        elif func(key, value):
            keys.append(sub.copy())
            values.append(value)
        if sub: sub.pop()

    keys = []
    values = []
    recursion(data, key=None, sub=[])
    return keys, values


def get_value(data, key, default=None):
    """根据key路径获得对应的值"""
    value = toolz.get_in(key, data, default=default)
    return value


def get_values(data, keys):
    """根据keys路径获得对应的值"""
    values = [toolz.get_in(k, data) for k in keys]
    return values


def set_values(data, keys, values):
    """根据keys路径设置对应的值"""
    for k, v in zip(keys, values):
        data = toolz_assoc_in(data, keys=k, value=v)
        # data = toolz.assoc_in(data, keys=k, value=v)
    return data


def set_value(data, key, value):
    """根据keys路径设置对应的值"""
    # content = toolz_assoc_in(content, keys=key, value=value)
    data = toolz.assoc_in(data, keys=key, value=value)
    return data


def del_key(data: dict, key: list):
    """
    安全删除嵌套字典的深层键
    :param data:
    :param key:
    :return:
    """
    out = data
    for k in key[:-1]:
        if not isinstance(out, dict) or k not in out:
            return None
        out = out[k]
    return out.pop(key[-1], None)


def del_keys(data: dict, keys: list):
    out = []
    for key in keys:
        v = del_key(data, key)
        out.append(v)
    return out


def toolz_assoc_in(data, keys, value):
    """toolz_assoc_in用来代替toolz.assoc_in"""
    cur_keys = []
    for i, k in enumerate(keys):
        if isinstance(k, str):
            cur_keys.append(k)
        elif isinstance(k, int):
            curObj = toolz.get_in(cur_keys + [k], data)
            if curObj == None:
                print("发现非法参数:obj:{}, keys:{}".format(toolz.get_in(cur_keys, data), keys))
                raise Exception("给定路径非法")
            newKeys = keys[i + 1:]
            if len(newKeys) == 0:
                toolz.get_in(cur_keys, data)[k] = value
            else:
                newValue = toolz_assoc_in(curObj, newKeys, value)
                toolz.get_in(cur_keys, data)[k] = newValue
            return data
    if len(cur_keys) == len(keys):
        return toolz.assoc_in(data, cur_keys, value)


if __name__ == "__main__":
    data = {'C': 0, 'A': 5, 'B': 3, 'D': 2}
    print(dict_sort(data))
