# -*- coding: utf-8 -*-
import logging
import time
import requests
import json
from pybaseutils import log_utils

logger = log_utils.get_logger()


def post(url, params=None, timeout=6, max_retries=1, **kwargs):
    """
    :param url: 请求URL
    :param params: 请求参数
    :param timeout: 设置超时
    :return:
    """
    result = None
    counts = 0
    while counts < max_retries:
        try:
            t1 = time.time()
            # r = requests.get(url, params=params, timeout=timeout,**kwargs)
            r = requests.post(url, json=params, timeout=timeout, **kwargs)
            t2 = time.time()
            elapsed = (t2 - t1) * 1000
            code = r.status_code
            if code == 200:
                logger.info(f'code={code}, url={url}, elapsed:{elapsed:3.3f}ms')
                result = r.json()
                break
            else:
                r.raise_for_status()  # 如果响应状态码不是200，抛出异常
        except Exception as e:
            counts += 1
            logger.error(f'Error msg:{e}, try to retry times={counts}/{max_retries}')
            time.sleep(0.1)
    return result


def get(url, params=None, timeout=6, max_retries=1, **kwargs):
    """
    :param url: 请求URL
    :param params: 请求参数
    :param timeout: 设置超时
    :return:
    """
    result = None
    counts = 0
    while counts < max_retries:
        try:
            t1 = time.time()
            r = requests.get(url, params=params, timeout=timeout, **kwargs)
            # r = requests.post(url, json=params, timeout=timeout, **kwargs)
            t2 = time.time()
            elapsed = (t2 - t1) * 1000
            code = r.status_code
            if code == 200:
                logger.info(f'code={code}, url={url}, elapsed:{elapsed:3.3f}ms')
                result = r.json()
                break
            else:
                r.raise_for_status()  # 如果响应状态码不是200，抛出异常
        except Exception as e:
            counts += 1
            logger.error(f'Error msg:{e}, try to retry times={counts}/{max_retries}')
            time.sleep(0.1)
    return result


def tojson(data: dict, keys: list):
    """
    指定需要反序列的数据
    :param data:
    :param keys:
    :return:
    """
    if isinstance(data, dict):
        for k, v in data.items():
            if k in keys:
                try:
                    data[k] = json.loads(v)
                except Exception as e:
                    print(e)
            else:
                data[k] = tojson(v, keys=keys)
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = tojson(data[i], keys=keys)
    return data
