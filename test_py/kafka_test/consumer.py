# -*-coding: utf-8 -*-
"""
    @Author :
    @E-mail :
    @Date   : 2023-05-29 17:41:51
    @Brief  : 接收数据
"""

import os
import argparse
import traceback
import json
import uuid
from kafka import KafkaConsumer
from pybaseutils import json_utils

# topic = 'aije-human-attribute-result-producer'
topic = 'aije-human-attribute-result'


class Consumer(object):
    def __init__(self, env, topic):
        self.env = env
        self.servers = self.env2servers(env=env)
        self.topic = topic
        print("env    :{}".format(self.env))
        print("servers:{}".format(self.servers))
        print("topic  :{}".format(self.topic))
        self.consumer = KafkaConsumer(topic,
                                      group_id='daip-calligraphy-single-word-consumer-group',
                                      bootstrap_servers=self.servers,
                                      value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                                      auto_offset_reset='earliest',
                                      fetch_max_wait_ms=1000,
                                      max_poll_interval_ms=300000,
                                      session_timeout_ms=10000,
                                      enable_auto_commit=True,
                                      auto_commit_interval_ms=1000,
                                      fetch_max_bytes=1024 * 100, )

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

    def task(self, output=""):
        nums = 0
        print("{}等待接受数据中...".format(self.env))
        for msg in self.consumer:
            try:
                data = msg.value
                name = uuid.uuid4().hex
                nums += 1
                print('{}'.format(json_utils.formatting(data)))
                print('{}收到第{}条消息'.format(self.env, nums, json_utils.formatting(data)))
                print("--------" * 10)
            except:
                traceback.print_exc()


def parser_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', type=str, default='dev', help='env to send',
                        choices=['local', 'dev', 'stage', 'test'])
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser_argument()
    p = Consumer(env=args.env, topic=topic)
    p.task(output="")
