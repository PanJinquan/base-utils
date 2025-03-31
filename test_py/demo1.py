import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from pybaseutils import file_utils, json_utils
import glob

# 使用示例
if __name__ == "__main__":
    data_root = "http://aije-mvp-nginx.partner.dm-ai.com/req-resp/aije-job-m8ch701s-awbg/*.json"
    # data_root = "/media/PKing/新加卷/SDK/base-utils/data/person/*.json"
    file = glob.glob(data_root)
    print(file)
    data =json_utils.read_json_data("http://aije-mvp-nginx.partner.dm-ai.com/req-resp/aije-job-m8ch701s-awbg/nlp-12.json")
    print(data)
