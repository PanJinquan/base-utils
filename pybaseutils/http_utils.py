# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 390737991@qq.com
# @Date   : 2019-12-31 09:11:25
# --------------------------------------------------------
"""
import os
import time
import requests
import json
from pybaseutils import log_utils, text_utils, json_utils, thread_utils

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


def get_url_files(url, prefix="", postfix=None, basename=False, timeout=30):
    """
    获得文件服务器中文件列表
    :param url: http URL地址等
    :param prefix: http URL地址等
    :param postfix: None或者[]表示所有文件
    :param basename: 返回的列表是文件名（True），还是文件的完整路径(False)
    :return:
    """
    from bs4 import BeautifulSoup
    try:
        if url.endswith("/"): url = url[:-1]
        # 设置超时和重试
        r = requests.get(url, timeout=timeout, auth=("admin", "admin"))
        r.raise_for_status()
        # 解析目录列表
        soup = BeautifulSoup(r.text, 'html.parser')
        file_list = [a['href'] for a in soup.select('a[href]')]
        # 过滤并拼接完整URL
        file_list = text_utils.find_match_texts(file_list, pattern=[prefix], org=True) if prefix else file_list
        file_list = text_utils.find_match_texts(file_list, pattern=postfix, org=True) if postfix else file_list
        file_list.sort()
        file_list = file_list if basename else [f"{url}/{f}" for f in file_list]
        return file_list
    except Exception as e:
        print(f"请求失败: {str(e)}")
        return []


def download_file(url, out, timeout=30, log=False):
    """
    根据url下载文件
    :param url: 文件URL
    :param out: 输出保存目录
    :param log: 是否打印LOG信息
    :return:
    """
    try:
        os.makedirs(out, exist_ok=True)
        name = url.split('/')[-1]
        path = os.path.join(out, name)
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.text
        if name.endswith("json"):
            json_utils.write_json_path(path, json_utils.str2dict(data=data))
        else:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(data)
        if log: print(f"下载文件: {url} --> {path}")
        return path
    except Exception as e:
        print(f"下载失败: {url}，{str(e)}")
        return None


def download_files(urls: list, out, max_workers=8):
    """
    :param urls: 文件URL列表
    :param out: 下载保持目录
    :param log: 下载保持目录
    :return: file_list, loss_list
    """
    pool = thread_utils.ThreadPool(max_workers=max_workers)
    inputs = [(f, out) for f in urls]
    t0 = time.time()
    output = pool.task_maps(func=download_file, inputs=inputs)
    t1 = time.time()
    dt = (t1 - t0) * 1000
    file_list, loss_list = [], []
    for i, file in enumerate(output):
        if file:
            file_list.append(file)
        else:
            loss_list.append(urls[i])
    print(f"启动{max_workers}个线程下载文件,成功:{len(file_list)},失败:{len(loss_list)},耗时:{dt:.2f}ms")
    return file_list, loss_list


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
