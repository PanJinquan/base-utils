import os
from time import sleep

import requests
import json
import time
from bs4 import BeautifulSoup
from tqdm import tqdm
from pybaseutils import http_utils

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


def search_images(keyword, output, prefix, max_nums=50, timeout=3):
    # 构建搜索URL
    search_url = "https://cn.bing.com/images/search"
    pages = 0
    count = 0
    print(f"开始搜索关键词: {keyword}")
    while count < max_nums:
        try:
            # 发送搜索请求
            res = requests.get(search_url, params={'q': keyword, 'first': pages, 'count': 20}, headers=headers)
            # 解析页面
            soup = BeautifulSoup(res.text, 'html.parser')
            # 提取图片链接
            for img in soup.find_all('a', class_='iusc'):
                try:
                    m = json.loads(img.get('m'))
                    url = m.get('murl', '')
                    name = url.split('/')[-1]  # 文件名
                    exts = name.split(".")[-1]  # 扩展名
                    name = f"{prefix}_{count + 1 :0=4d}.{exts}"  # 文件名
                    path = os.path.join(output, name)  # 保存路径
                    if exts not in ["jpg", "png", "jpeg"]: continue
                    if http_utils.download_file(url, path, timeout=timeout): count += 1
                    time.sleep(0.5)
                except Exception as e:
                    print(f"下载图片失败: {str(e)}")
            pages += 20
        except Exception as e:
            print(f"搜索出错: {str(e)}")
            break
    print(f"\n下载完成! 共下载 {count} 张图片")


if __name__ == "__main__":
    keywords = ['低头人脸照片', '闭眼人脸照片', '张嘴人脸照片', "侧脸人脸照片"]
    output = "/home/PKing/Downloads/search_images/biying"
    for i, word in tqdm(enumerate(keywords)):
        keyword = f"{word}"
        prefix = f"image3_{i :0=4d}"
        search_images(keyword, output=os.path.join(output, word), prefix=prefix, max_nums=200)
