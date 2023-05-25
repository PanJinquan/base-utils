# -*- coding:utf-8 -*-
"""
消费kafka数据,并推送到kafka
"""
import queue
import traceback
import logging
import json
import time
from typing import Callable, List, Tuple, Dict


class KafkaWorker(object):

    def __init__(self):
        pass

    @staticmethod
    def handle_task1(msg: Dict):
        data: Dict = msg.pop("data")
        result = {}
        data.update({"result": result})
        msg['data'] = data
        return msg


if __name__ == '__main__':
    msg = {"data": {"image": "image-data"}}
    msg = KafkaWorker().handle_task1(msg)
    print(msg)
