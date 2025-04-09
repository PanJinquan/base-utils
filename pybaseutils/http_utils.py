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
            logger.error(f'Error msg:{e}, url={url}, try to retry times={counts}/{max_retries}')
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
    log = kwargs.pop("log", True)
    while counts < max_retries:
        try:
            t1 = time.time()
            r = requests.get(url, params=params, timeout=timeout, **kwargs)
            # r = requests.post(url, json=params, timeout=timeout, **kwargs)
            t2 = time.time()
            elapsed = (t2 - t1) * 1000
            code = r.status_code
            if code == 200:
                if log: logger.info(f'code={code}, url={url}, elapsed:{elapsed:3.3f}ms')
                result = r.json()
                break
            else:
                r.raise_for_status()  # 如果响应状态码不是200，抛出异常
        except Exception as e:
            counts += 1
            logger.error(f'Error msg:{e}, url={url}, try to retry times={counts}/{max_retries}')
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


def download_file(url, out, timeout=30, max_retries=1, log=False):
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
        data = get(url, params=None, timeout=timeout, max_retries=max_retries, log=False)
        assert data
        if name.endswith("json"):
            json_utils.save_json(path, data)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(data)
        if log: print(f"下载成功: {url} --> {path}")
        return path
    except Exception as e:
        if log: print(f"下载失败: {url}")
        return None


def download_files(urls: list, out, max_workers=8, timeout=30, max_retries=1):
    """
    :param urls: 文件URL列表
    :param out: 下载保持目录
    :param max_workers:
    :param timeout:
    :param max_retries:
    :return: 返回下载成功的本地文件列表file_list
    """
    pool = thread_utils.ThreadPool(max_workers=max_workers)
    inputs = [(url, out, timeout, max_retries) for url in urls]
    print("----" * 10)
    print(f"启动{max_workers}个线程下载{len(urls)}个文件,请等待....")
    t0 = time.time()
    file_list = pool.task_maps(func=download_file, inputs=inputs)
    t1 = time.time()
    dt = (t1 - t0) * 1000
    # 下载失败的url列表loss_list
    loss_list = [urls[i] for i in range(len(file_list)) if not file_list[i]]
    print(f"成功:{len(file_list) - len(loss_list)},失败:{len(loss_list)},耗时:{dt:.2f}ms")
    return file_list, loss_list


def read_url_file(url, timeout=30, max_retries=1):
    """
    根据url读取文件
    :param url: 文件URL
    :param timeout:
    :param max_retries:
    :return:
    """
    try:
        data = get(url, params=None, timeout=timeout, max_retries=max_retries, log=False)
        return data
    except Exception as e:
        print(f"读取失败: {url}")
        return None


def read_url_files(urls: list, max_workers=8, timeout=30, max_retries=1):
    """
    根据url读取文件
    :param urls: 文件urls列表
    :param max_workers: 开启线程数目
    :param timeout:
    :param max_retries:
    :return: 返回读取成功的文件数据file_list
    """
    pool = thread_utils.ThreadPool(max_workers=max_workers)
    inputs = [(url, timeout, max_retries) for url in urls]
    print("----" * 10)
    print(f"启动{max_workers}个线程读取{len(urls)}个文件,请等待....")
    t0 = time.time()
    file_list = pool.task_maps(func=read_url_file, inputs=inputs)
    t1 = time.time()
    dt = (t1 - t0) * 1000
    # 读取失败的url列表loss_list
    loss_list = [urls[i] for i in range(len(file_list)) if not file_list[i]]
    print(f"成功:{len(file_list) - len(loss_list)},失败:{len(loss_list)},耗时:{dt:.2f}ms")
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


if __name__ == '__main__':
    urls = [
        "http://aije-mvp-nginx.partner.dm-ai.com/req-resp/aije-job-m8ch701s-awbg/nlp-12.json",
        "http://aije-mvp-nginx.partner.dm-ai.com/req-resp/aije-job-m8ch701s-awbg/nlp-13.json",
        "http://aije-mvp-nginx.partner.dm-ai.com/req-resp/aije-job-m8ch701s-awbg/nlp-17.json",
    ]
    out = "/home/PKing/Downloads/tmp"
    url = 'https://aije-mvp-nginx.partner.dm-ai.com/req-resp/aije-job-m911dzow-17cu'
    urls = get_url_files(url, postfix=["*.json"])
    file_list1, loss_list1 = download_files(urls, out=out, max_retries=1)
    file_list2, loss_list2 = read_url_files(urls, max_retries=1)
    print(loss_list1)
    print(loss_list2)
