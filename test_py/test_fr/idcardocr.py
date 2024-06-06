# -*-coding: utf-8 -*-
"""
    @Author :
    @E-mail :
    @Date   : 2023-05-29 17:41:51
    @Brief  : API接口测试
"""

import os
import argparse
import json
import cv2
import requests
import base64
import uuid
import threading
import time
import copy
import traceback
from datetime import datetime
import hmac
import hashlib
from pybaseutils import json_utils, thread_utils, image_utils, file_utils


class MockRequest(object):
    def __init__(self):
        self.URL = "http://172.16.128.139:7002/d5b72d5b/youtu/ocrapi/idcardocr"  # 身份证识别
        self.URI = "/d5b72d5b/youtu/ocrapi/idcardocr"
        self.AK = "f7c5e573"  # accessId
        self.SK = "430b5e006c"
        self.APP_ID = "20220421000001"
        self.time = str(time.time() // 1000)
        self.getAccessToken(self.URI, self.AK, self.SK, self.APP_ID, self.time)

    def getAccessToken(self, uri, accessId, accessSecret, useId, time):
        """
        获取访问令牌的加密算法
        :param uri:
        :param accessId:
        :param accessSecret:
        :param useId:
        :param time:
        :return:
        """
        one = uri + accessId + useId + "#" + time
        encoded = base64.b64encode(one.encode('utf-8'))
        two = encoded.decode('utf-8')
        # 密钥
        secret_key = accessSecret
        # 使用hmac模块和hashlib库计算HMAC-SHA256
        digest = hmac.new(secret_key.encode('utf-8'), msg=two.encode('utf-8'), digestmod=hashlib.sha256).digest()
        encoded = base64.b64encode(digest)
        three = encoded.decode('utf-8').upper()
        return three

    def post_request(self, params=None):
        t1 = time.time()
        r = requests.post(self.URL, json=params, headers={
            'applicationId': self.APP_ID,  # appId
            'accessId': self.AK,  # AK
            'accessToken': self.getAccessToken(self.URI, self.AK, self.SK, self.APP_ID, self.time),  # 通过SK生成
            'time': self.time,
            "content-type": "application/json"
        })
        t2 = time.time()
        code = r.status_code
        print("url: {} code:{},response elapsed: {:3.3f}ms".format(self.url, code, (t2 - t1) * 1000, ))
        if code == 200:
            r = r.json()
            self.print_info(r)
            print("----" * 20)
        else:
            print(r)
            assert code == 200
        return r

    def request(self, image_file):
        """
        :param image_bs64:
        :return:
        """
        params = {
            "app_id": self.APP_ID,
            "image": image_utils.read_image_base64(image_file),
            "session_id": "SESSION_ID",
            "card_type": 2,
            "ret_image": False,
        }
        r = self.post_request(params=params)
        return r

    @staticmethod
    def print_info(r):
        info = json.dumps(r, indent=1, separators=(', ', ': '), ensure_ascii=False)
        print(info)
        json_utils.write_json_path("result.json", r)


if __name__ == '__main__':
    service = MockRequest()
    file = "img.png"
    service.request(file)
