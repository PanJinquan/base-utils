import cv2
from pybaseutils import file_utils, image_utils,json_utils

import toolz


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


def del_keys(data, keys: list):
    out = []
    for key in keys:
        v = del_key(data, key)
        out.append(v)
    return out


