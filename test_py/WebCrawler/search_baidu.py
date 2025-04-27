import requests
import os
import re
import time
import urllib.parse
from tqdm import tqdm
from pybaseutils import http_utils

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


def search_images(keyword, output, prefix, max_nums=50, timeout=3):
    encoded_keyword = urllib.parse.quote(keyword)
    print(f"开始搜索关键词: {keyword}")
    pages = 0
    count = 0
    while count < max_nums:
        search_url = f"https://image.baidu.com/search/flip?tn=baiduimage&word={encoded_keyword}&pn={pages * 30}"
        try:
            r = requests.get(search_url, headers=headers, timeout=10)
            r.encoding = 'utf-8'
            # 使用正则表达式提取图片URL
            links = re.findall('"objURL":"(.*?)"', r.text)
            if not links:
                print("没有找到更多图片")
                break
            for url in links:
                if count >= max_nums: break
                try:
                    name = url.split('/')[-1]  # 文件名
                    exts = name.split(".")[-1]  # 扩展名
                    name = f"{prefix}_{count + 1 :0=4d}.{exts}"  # 文件名
                    path = os.path.join(output, name)  # 保存路径
                    if exts not in ["jpg", "png", "jpeg"]: continue
                    if http_utils.download_file(url, path, timeout=timeout): count += 1
                    time.sleep(0.5)
                except Exception as e:
                    print(f"下载图片失败: {str(e)}")
            pages += 1
        except Exception as e:
            print(f"搜索页面出错: {str(e)}")
            break

    print(f"\n下载完成! 共下载 {count} 张图片")


if __name__ == "__main__":
    keywords = ['低头人脸照片', '闭眼人脸照片', '张嘴人脸照片', "侧脸人脸照片"]
    output = "/home/PKing/Downloads/search_images/baidu"
    for i, word in tqdm(enumerate(keywords)):
        keyword = f"{word}"
        prefix = f"image4_{i :0=4d}"
        search_images(keyword, output=os.path.join(output, word), prefix=prefix, max_nums=200)
