# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  :
"""
import numpy as np
import cv2, threading, time
import socket


class MediaService(object):
    """流媒体客户端"""

    def __init__(self, service, port=8080, size=(640, 480)):
        """
        :param service: 流媒体服务地址
        :param port:端口
        :param size: 图像大小(H,W)
        """
        self.address = (service, port)
        self.size = size
        self.bufsize = self.size[0] * self.size[1] * 3
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # self.socket.bind(self.address)

    def send(self):
        """
        发生数据
        """
        cap = cv2.VideoCapture(0)
        while True:
            _, image = cap.read()
            image = cv2.flip(image, 1)
            # 图片太大，会导致Message too long的异常
            # _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 50])
            _, buffer = cv2.imencode('.jpg', image)
            t0 = time.time()
            self.socket.sendto(buffer, self.address)
            t1 = time.time()
            elapsed = (t1 - t0) * 1000
            print('已发送{}Bytes的数据,elapsed:{:3.3f}ms'.format(image.size, elapsed))
            cv2.imshow('service', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    service = "192.168.2.219"
    c = MediaService(service=service, size=(640, 480))
    c.send()
