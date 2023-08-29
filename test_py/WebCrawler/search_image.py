import requests
import re
import os
import time
from datetime import datetime


class WebCrawler(object):
    def __init__(self, output="./output"):
        self.headers = {  # 文件头，必须有，否则会安全验证
            "Accept": "application/json, text/javascript, */*; q=0.01",
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'Connection': 'keep-alive',
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36",
            'Host': 'image.baidu.com',
            'Referer': 'https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=&st=-1&fm=result&fr=&sf=1&fmq=1610952036123_R&pv=&ic=&nc=1&z=&hd=&latest=&copyright=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&sid=&word=%E6%98%9F%E9%99%85',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'X-Requested-With': 'XMLHttpRequest'
        }
        self.url = 'http://image.baidu.com/search/index?tn=baiduimage&fm=result&ie=utf-8&word='
        self.output = output

    def search(self, keyword="老鹰", countmax=10):
        """
        :param keyword:
        :param countmax:
        :return:
        """
        url = self.url + keyword + "&pn="
        output = os.path.join(self.output, keyword)
        if not os.path.exists(output): os.makedirs(output)
        strhtml = requests.get(url, headers=self.headers)  # get方式获取数据
        string = str(strhtml.text)
        # 正则表达式取得图片总数量
        totalnum = re.findall('<div id="resultInfo" style="font-size: 13px;">(.*?)</div>', string)
        print("百度图片" + totalnum[0])
        img_url_regex = '"thumbURL":"(.*?)",'  # 正则匹配式
        count = 0  # 总共下载的图片数
        index = 0  # 链接后面的序号
        page = 0  # 当前搜集的页
        while True:
            strhtml = requests.get(url + str(index), headers=self.headers)  # get方式获取数据
            string = str(strhtml.text)
            print("已爬取网页")
            pic_url = re.findall(img_url_regex, string)  # 先利用正则表达式找到图片url
            print("第" + str(page + 1) + "页共收集到" + str(len(pic_url)) + "张图片")
            index += len(pic_url)  # 网址索引向后，跳到下一页继续搜刮图片
            for each in pic_url:
                print('正在下载第' + str(count + 1) + '张图片，图片地址:' + str(each))
                try:
                    if each is not None:
                        pic = requests.get(each, timeout=5)
                    else:
                        continue
                except BaseException:
                    print('错误，当前图片无法下载')
                    continue
                else:
                    time = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S_%f')
                    image_file = os.path.join(output, "{}_{:3=d}.jpg".format(time, count))
                    fp = open(image_file, 'wb')
                    fp.write(pic.content)
                    fp.close()
                    count += 1
                if count >= countmax:
                    break
            if count >= countmax:
                break


if __name__ == '__main__':
    output = "/home/PKing/nasdata/dataset/tmp/challenge/鸟类品种识别/鸟类品种识别挑战赛训练集/"
    keyword = "老鹰"
    countmax = 300
    web = WebCrawler(output)
    web.search(keyword=keyword, countmax=countmax)
