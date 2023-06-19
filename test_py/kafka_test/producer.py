# -*-coding: utf-8 -*-
"""
    @Author :
    @E-mail :
    @Date   : 2023-05-29 17:41:51
    @Brief  : 发送数据
"""

import os
import cv2
import uuid
import json
import argparse
import threading
from kafka import KafkaProducer
from pybaseutils import file_utils, base64_utils, thread_utils

topic = 'aije-human-attribute-consumer'


class Producer(object):
    def __init__(self, env, topic):
        self.env = env
        self.servers = self.env2servers(env=env)
        self.topic = topic
        print("env    :{}".format(self.env))
        print("servers:{}".format(self.servers))
        print("topic  :{}".format(self.topic))
        self.producer = KafkaProducer(bootstrap_servers=self.servers,
                                      retries=5, acks='all',
                                      value_serializer=lambda m: json.dumps(m, ensure_ascii=False).encode('utf-8'),
                                      max_in_flight_requests_per_connection=1,
                                      max_block_ms=5000,
                                      request_timeout_ms=2000,
                                      max_request_size=3173440261)

    @staticmethod
    def env2servers(env, app=""):
        env_servers = {
            "dev": ['dev01-public-kafka.dm-ai.com:9092', 'dev02-public-kafka.dm-ai.com:9092',
                    'dev03-public-kafka.dm-ai.com:9092'],  # dev
            "stage": ["192.168.3.88:9092", "192.168.3.89:9092", "192.168.3.90:9092"],  # stage
            "partner": ['10.12.15.11:9092', '10.12.15.12:9092', '10.12.15.13:9092'],  # partner
            "prd": ["192.168.11.59:9092", "192.168.12.59:9092", "192.168.13.59:9092"],  # partner
        }
        return env_servers[env]

    def get_inputs(self, image):
        data = {
            "metadata": {
                "task_id": "189a151c-5eda-41fa-9fc2-a953c3a04b19",
                "video_id": "11100002",
            },
            "data": {
                "image": image,
            },
            "id": "11100002_17100329_545_2610_718_2783_1651716233803"
        }
        return data

    def task(self, image_file):
        """
        发送单个数据
        :param image_file:
        :return:
        """
        print("sent image_file:{}".format(image_file))
        image = cv2.imread(image_file)
        data = self.get_inputs(image)
        data = base64_utils.serialization(data)
        result = self.producer.send(topic=self.topic, value=data, key=uuid.uuid4().hex.encode('utf-8'))
        self.producer.flush()
        return result

    def task_batch(self, image_dir):
        """
        批量串行发送数据
        :param image_dir:
        :return:
        """
        image_list = file_utils.get_images_list(image_dir)
        for i, file in enumerate(image_list):
            result = self.task(file)
            i += 1
            print("result:{}".format(result.get()))
            print('Producer:发送{}条消息'.format(i))

    def task_parallel(self, image_dir, max_workers=8):
        """
        并行发送数据
        :param image_dir:
        :param max_workers:
        :return:
        """
        image_list = file_utils.get_images_list(image_dir)
        image_list = image_list * 100
        t = thread_utils.ThreadPool(max_workers=max_workers)
        result = t.task_map(func=self.task, inputs=image_list)
        print('Producer:发送{}条消息'.format(len(image_list)))
        for r in result:
            print("result:{}".format(r.get()))


def parser_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', type=str, default='dev', help='env to send',
                        choices=['local', 'dev', 'stage', 'test'])
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    image_dir = "./data"
    args = parser_argument()
    p = Producer(env=args.env, topic=topic)
    for i in range(1):
        p.task_batch(image_dir)
        # p.task_parallel(image_dir)
