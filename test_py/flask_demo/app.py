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

app = flask.Flask(__name__, template_folder="templates", static_folder="static", root_path=None)


def task(filename):
    img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)
    cv2.imwrite("./static/cache/ldh_output.jpg", img)
    return img


@app.route('/', methods=['get'])
def index():
    return flask.render_template("index.html")


@app.route('/process', methods=['post'])
def process():
    file = flask.request.files['image_file']
    filename = file.filename
    print("upload file :{}".format(filename))
    savename = r'static/cache/' + file.filename
    file.save(savename)
    print("save file :{}".format(savename))
    image = task(savename)
    # image = cv2.imread(filename)
    print("image：", image.shape)
    data = image_utils.image2bytes(image)
    response = flask.make_response(data)
    response.headers['Content-Type'] = 'image/jpg'
    return response
    # return flask.render_template('result.html')
    # return flask.render_template('index.html')


if __name__ == '__main__':
    # app.run(host='127.0.0.1', port=5000, debug=False)
    app.run(host='0.0.0.0', port=5000, debug=False)
