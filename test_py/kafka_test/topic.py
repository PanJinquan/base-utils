# -*- coding: utf-8 -*-
"""
    @Author :
    @E-mail :
    @Date   : 2023-05-29 17:41:51
    @Brief  : 发送数据
"""
from kafka import KafkaAdminClient
from kafka.admin import NewTopic
from kafka.errors import TopicAlreadyExistsError

topic = 'xxx-consumer'


class kafkaInfo():
    def __init__(self, env):
        """
        :param env: 环境
        """
        self.servers = self.env2servers(env=env)
        self.admin = KafkaAdminClient(bootstrap_servers=self.servers)

    def create_topic(self, topic, num_partitions=1, replication_factor=1):
        """
        :param topic: 创建一个topic
        :param num_partitions: 指定分区数
        :param replication_factor:用来设置topic的副本数，每个topic可以有多个副本
        :return:
        """
        try:
            new_topic = NewTopic(topic, num_partitions=num_partitions, replication_factor=replication_factor)
            r = self.admin.create_topics([new_topic])
            print("create topic successfully: {}".format(topic))
        except TopicAlreadyExistsError as e:
            print(e.message)

    def delete_topic(self, topic):
        """
        :param topic: 删除一个topic
        :return:
        """
        # 删除topic
        self.admin.delete_topics([topic])

    def get_consumer_group(self):
        """
        显示所有的消费组
        :return:
        """
        print(self.admin.list_consumer_groups())
        # 显示消费组的offsets
        print(self.admin.list_consumer_group_offsets("kafka-group-id"))

    @staticmethod
    def env2servers(env):
        env_servers = {

        }
        return env_servers[env]


if __name__ == "__main__":
    k = kafkaInfo(env="dev")
    k.create_topic(topic=topic)
