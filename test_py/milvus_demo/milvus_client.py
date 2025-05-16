# -*- coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2024-10-29 11:26:34
    @Brief  :
"""
import time
from typing import List, Dict
from pymilvus import (db,
                      connections,
                      utility,
                      FieldSchema,
                      CollectionSchema,
                      DataType,
                      Collection,
                      )

# 连接到 Milvus 服务器
# connections.connect("default", uri="http://10.13.3.22:19530")
# connections.connect("default", uri="http://127.0.0.1:19530")
connections.connect("default", uri="http://192.168.2.52:19530")

db_name = "aije_algorithm_dev"


def create_database(db_name=db_name):
    """
    创建数据库
    :param db_name:
    :return:
    """
    if db_name not in db.list_database():
        db.create_database(db_name)  # 仅企业版支持
        print(f"Database '{db_name}' created")
    else:
        print(f"Database '{db_name}' already exists")
    # 切换数据库
    db.using_database(db_name)


class MilvusCollection(object):
    def __init__(self, col_name, dim, drop=False):
        """
        向量数据库集合
        :param col_name:集合名称
        :param dim: 数据特征维度(Embedding-Dim)
        :param drop: True,如果collection存在,则删除;
        """
        self.drop = drop
        self.col_name = col_name
        self.collection = self.create(col_name=self.col_name, dim=dim, drop=self.drop)

    @staticmethod
    def get_collections():
        """获得所有集合名称"""
        keys = utility.list_collections()
        print("collections:{}".format(keys))
        return keys

    @staticmethod
    def del_collections(keys=[]):
        """
        删除集合
        :param keys: 如keys=None,则表示删除所有集合
        :return:
        """
        if keys is None: keys = MilvusCollection.get_collections()
        for name in keys:
            if utility.has_collection(name):  # 删除原始的集合
                print("delete collections --> {}".format(name))
                utility.drop_collection(collection_name=name)
        return True

    @staticmethod
    def create(col_name, dim, drop=False) -> Collection:
        """
        :param col_name: 集合名称
        :param dim:  数据特征维度(Embedding-Dim)
        :param drop: True,如果collection存在,则删除;
        :return:
        """
        if drop and utility.has_collection(col_name):  # 删除原始的集合
            utility.drop_collection(collection_name=col_name)
        fields = [
            FieldSchema(name='id', dtype=DataType.INT64, descrition='embedding ID', is_primary=True, auto_id=True),
            FieldSchema(name='feature', dtype=DataType.FLOAT_VECTOR, descrition='embedding vectors', dim=dim),
            FieldSchema(name='info', dtype=DataType.JSON, descrition='data information', max_length=500),
        ]
        # 定义集合 Schema
        schema = CollectionSchema(fields=fields, description='CollectionSchema')
        collection = Collection(name=col_name, schema=schema)
        return collection

    def insert(self, inputs: List):
        """
        :param inputs: [field0,field1,field2,...]
        :return:
        """
        r = self.collection.insert(inputs)  # 插入数据
        return r

    def flush(self):
        # 插入数据后，调用 flush 方法将数据持久化到磁盘。
        self.collection.flush()

    def search(self, vectors: List, metric_type="IP", top_k=3, flush=True):
        """
        :param vectors: 查询向量 [v0,v1,v2]
        :param metric_type: L2，IP(必须归一化)
                          参考：https://milvus.io/docs/v2.2.x/metric.md?tab=floating
        :param top_k:
        :param flush: 查询前如果有插入操作，请先刷新
        :return:
        """
        if flush: self.flush()
        # 创建索引
        index_params = {
            "index_type": "IVF_FLAT",
            "params": {"nlist": 100},
            "metric_type": metric_type
        }
        self.collection.create_index(field_name="feature", index_params=index_params)
        # 加载集合
        self.collection.load()
        # 定义搜索参数
        params = {"metric_type": metric_type, "params": {"nprobe": 10}}
        # 执行搜索
        output = self.collection.search(data=vectors,
                                        anns_field="feature",
                                        output_fields=["id", "feature", "info"],
                                        param=params,
                                        limit=top_k,
                                        expr=None
                                        )
        # 返回结果
        results = []
        for hits in output:
            r = [dict(id=hit.id, score=hit.score, fields=hit.fields) for hit in hits]
            results.append(r)
        return results

    def print_results(self, results: List):
        print("----" * 20)
        for i, res in enumerate(results):
            for info in res:
                print(f"i={i:4d}  {info}")
            print("----" * 20)


if __name__ == '__main__':
    pass
