# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  :
"""
import os
import cv2
import time
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
from pybaseutils import file_utils, image_utils, base64_utils, time_utils

import socket
import cv2
import numpy
from time import sleep

# socket.AF_INET 用于服务器与服务器之间的网络通信
# socket.SOCK_STREAM 代表基于TCP的流式socket通信
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 连接服务端
address_server = ('172.17.76.32', 8888)
sock.connect(address_server)

# 从摄像头采集图像
# 参数是0,表示打开笔记本的内置摄像头,参数是视频文件路径则打开视频
capture = cv2.VideoCapture(0)
# capture.read() 按帧读取视频
# ret,frame 是capture.read()方法的返回值q
# 其中ret是布尔值，如果读取帧正确，返回True;如果文件读到末尾，返回False。
# frame 就是每一帧图像，是个三维矩阵
ret, frame = capture.read()
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
while ret:
    # 首先对图片进行编码，因为socket不支持直接发送图片
    # '.jpg'表示把当前图片frame按照jpg格式编码
    # result, img_encode = cv2.imencode('.jpg', frame)
    img_encode = cv2.imencode('.jpg', frame, encode_param)[1]
    # result, img_encode = cv2.imencode('.jpg', frame)
    data = numpy.array(img_encode)
    stringData = data.tostring()
    # stringData = img_encode.tobytes()
    # 首先发送图片编码后的长度
    sock.send(str(len(stringData)).ljust(16).encode())
    # 然后一个字节一个字节发送编码的内容
    # 如果是python对python那么可以一次性发送，如果发给c++的server则必须分开发因为编码里面有字符串结束标志位，c++会截断
    # for i in range(0, len(stringData)):
    #     sock.send(stringData[i])
    sock.send(stringData)
    # sleep(1)
    print(str(len(stringData)).ljust(16).encode() + (stringData))
    ret, frame = capture.read()
    cv2.resize(frame, (640, 480))
sock.close()
cv2.destroyAllWindows()

if __name__ == "__main__":
    image_file = "test.png"
    image = cv2.imread(image_file)
    image_utils.cv_show_image("image",image)
