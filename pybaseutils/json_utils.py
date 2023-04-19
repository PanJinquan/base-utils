# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-04-19 10:40:26
    @Brief  :
"""
import os
import toolz
import json
import numbers
from pybaseutils.file_utils import read_json_data, write_json_path
from typing import List, Tuple, Dict


class Dict2Obj(object):
    """ dict转类对象"""

    def __init__(self, args):
        self.__dict__.update(args)


def formatting(content):
    """格式化json数据"""
    info = json.dumps(content, indent=1, separators=(', ', ': '), ensure_ascii=False)
    return info


def get_keys_vaules(content, func=None):
    """
    遍历json数据并获得所有value的key路径
    :param content:
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
    recursion(content, key=None, sub=[])
    return keys, values


def get_value(content, key, default=None):
    """根据key路径获得对应的值"""
    value = toolz.get_in(key, content, default=default)
    return value


def get_values(content, keys):
    """根据keys路径获得对应的值"""
    values = [toolz.get_in(k, content) for k in keys]
    return values


def set_values(content, keys, values):
    """根据keys路径设置对应的值"""
    for k, v in zip(keys, values):
        content = toolz_assoc_in(content, keys=k, value=v)
        # data = toolz.assoc_in(data, keys=k, value=v)
    return content


def toolz_assoc_in(content, keys, value):
    """toolz_assoc_in用来代替toolz.assoc_in"""
    cur_keys = []
    for i, k in enumerate(keys):
        if isinstance(k, str):
            cur_keys.append(k)
        elif isinstance(k, int):
            curObj = toolz.get_in(cur_keys + [k], content)
            if curObj == None:
                print("发现非法参数:obj:{}, keys:{}".format(toolz.get_in(cur_keys, content), keys))
                raise Exception("给定路径非法")
            newKeys = keys[i + 1:]
            if len(newKeys) == 0:
                toolz.get_in(cur_keys, content)[k] = value
            else:
                newValue = toolz_assoc_in(curObj, newKeys, value)
                toolz.get_in(cur_keys, content)[k] = newValue
            return content
    if len(cur_keys) == len(keys):
        return toolz.assoc_in(content, cur_keys, value)


if __name__ == "__main__":
    content = {
        "code": "0",
        "data": {
            "image": ["image1", 0],
            "file": {"file1": "path/to/image1.jpg", "file2": 2, "file3": ["file3_v1", "file3_v2"], },
            "url": "url1"
        }
    }
    func = lambda k, v: isinstance(v, numbers.Number)
    # 遍历获得data中所有value的路径
    keys, values = get_keys_vaules(content, func=func)
    for k, v in zip(keys, values):
        print("path={}\t    value={}".format(k, v))
    print("===" * 20)
    # toolz使用toolz工具或得所有keys的值,values1与values的值是一样的
    # values1 = get_values(content, keys=keys)
    # values1 = get_values(content, keys=[['data1', 'image', 1], ['data', 'file', 'file11']])
    values1 = get_value(None, key=['data', 'image', 1],default={"data"})
    print(values1)
    print("===" * 20)
    values = list(range(len(values)))
    content = set_values(content, keys=keys, values=values)
    print(formatting(content))
    print("===" * 20)
