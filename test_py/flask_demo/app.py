# -*- coding: utf-8 -*-
"""
    @Author : 
    @E-mail : 
    @Date   : 2023-05-16 08:53:11
    @Brief  : https://www.runoob.com/flask/flask-tutorial.html
"""
import os
import sys
import cv2
import numpy as np
import flask
import base64
from pybaseutils import image_utils, file_utils
from utils import utils

app = flask.Flask(__name__, template_folder="templates", static_folder="static", root_path=None)
ui = "index.html"
ptr = 0
cache_file = []
cache_root = "static/cache"


def task(image):
    image = cv2.GaussianBlur(image, (15, 15), 0)
    return image


@app.route('/', methods=['get'])
def index():
    return flask.render_template(ui)


@app.route('/process', methods=['post'])
def process():
    global ptr
    try:
        # file = flask.request.files['image_file'] # 单个文件
        files = flask.request.files.getlist('image_file')  # 多个文件
        temps = utils.savefiles(cache_root, files)
        cache_file.extend(temps)
        if not cache_file: return flask.render_template(ui)
        if temps: ptr = len(cache_file) - len(temps)
        path = cache_file[ptr]
        # src = utils.bytes2image(files[0].read())
        src = image_utils.read_image(path)
        dst = task(src)
        name = os.path.basename(path)
        src_info = {"code": 0, "msg": "success", "file": name}
        dst_info = {"code": 0, "msg": "success", "file": name}
        count_info = "[{}/{}]正在处理,file={}".format(ptr + 1, len(cache_file), name)
        print(count_info)
        # 将处理后的图像数据编码为base64,将编码后的图像数据传递给模板
        r = flask.render_template(ui,
                                  src_image=image_utils.image2base64(src),
                                  src_info=src_info,
                                  dst_image=image_utils.image2base64(dst),
                                  dst_info=dst_info,
                                  count=count_info
                                  )
    except Exception as e:
        print("Error：cache nums:{}/{},请求参数异常：".format(ptr, len(cache_file)), e)
        r = flask.render_template(ui)
    return r


@app.route('/next', methods=['post'])
def next():
    global ptr
    ptr = max(0, min(len(cache_file) - 1, ptr + 1))
    return process()


@app.route('/last', methods=['post'])
def last():
    global ptr
    ptr = max(0, ptr - 1)
    return process()


@app.route('/clear', methods=['post'])
def clear():
    global ptr, cache_file
    ptr = 0
    cache_file = []
    file_utils.remove_dir(cache_root)
    print("清空所有数据...")
    return process()


if __name__ == '__main__':
    # app.run(host='127.0.0.1', port=5000, debug=False)
    app.run(host='0.0.0.0', port=5000, debug=True)
