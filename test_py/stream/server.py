# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  :
"""
import sys, os
import numpy as np
import cv2, threading, time
import socket


class MediaClient(object):
    """流媒体客户端"""

    def __init__(self, service, port=8080, size=(640, 480)):
        """
        :param service: 流媒体服务地址
        :param port:端口
        :param size: 图像大小(H,W)
        """
        self.size = size
        self.bufsize = self.size[0] * self.size[1] * 3
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.address = (service, port)
        self.socket.bind(self.address)
        print("start socket service:{}，size={}".format(self.address, self.size))

    def read(self):
        """
        一般情况下，send()/recv()用于TCP协议下网络I/O操作;
        sendto()/recvfrom()用于UDP协议下网络I/O操作，
        但是如果在TCP中connect函数调用后，它们也可用于TCP传输。
        :return:
        """
        while True:
            buffer, _ = self.socket.recvfrom(self.bufsize)
            data = np.frombuffer(buffer, dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            image = self.task(image)
            cv2.imshow('Client', image)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cv2.destroyAllWindows()

    def task(self, image):
        return image


if __name__ == "__main__":
    service = "192.168.2.219"
    c = MediaClient(service=service, size=(640, 480))
    c.read()
