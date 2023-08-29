import requests  # 爬虫库
import re  # 正则表达式库
import os  # 系统库
import time  # 时间库
from datetime import datetime

headers = {  # 文件头，必须有，否则会安全验证
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

save_dir = "./BAIDU"
url = 'http://image.baidu.com/search/index?tn=baiduimage&fm=result&ie=utf-8&word='  # 百度链接
keyword = "老鹰"  # 请输入图片关键词
countmax = 50  # 请输入要爬取的图片数量
url = url + keyword + "&pn="
time_start = time.time()  # 获取初始时间
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
strhtml = requests.get(url, headers=headers)  # get方式获取数据
string = str(strhtml.text)
# with open("data.txt","w",encoding='utf-8') as f:#这个编码是个问题
#     f.write(string)  #这句话自带文件关闭功能，不需要再写f.close()
# print("已爬取，数据存入data.txt")

# 正则表达式取得图片总数量
totalnum = re.findall('<div id="resultInfo" style="font-size: 13px;">(.*?)</div>', string)
print("百度图片" + totalnum[0])

img_url_regex = '"thumbURL":"(.*?)",'  # 正则匹配式
count = 0  # 总共下载的图片数
index = 0  # 链接后面的序号
page = 0  # 当前搜集的页
while (1):
    strhtml = requests.get(url + str(index), headers=headers)  # get方式获取数据
    string = str(strhtml.text)
    print("已爬取网页")
    pic_url = re.findall(img_url_regex, string)  # 先利用正则表达式找到图片url
    print("第" + str(page + 1) + "页共收集到" + str(len(pic_url)) + "张图片")
    index += len(pic_url)  # 网址索引向后，跳到下一页继续搜刮图片
    try:  # 如果没有文件夹就创建
        os.mkdir('.' + r'\\' + keyword)
    except:
        pass

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
            image_path = os.path.join(save_dir, "{}_{:3=d}.jpg".format(time, count))
            fp = open(image_path, 'wb')
            fp.write(pic.content)
            fp.close()
            count += 1
        if countmax == count:
            break
    if countmax == count:
        break
