# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2024-10-29 11:26:34
    @Brief  :
"""
from typing import List, Dict
from pymilvus import (connections,
                      utility,
                      FieldSchema,
                      CollectionSchema,
                      DataType,
                      Collection,
                      )

# 连接到 Milvus 服务器
connections.connect("default", uri="http://10.13.3.22:19530", db_name="aije_algorithm_dev")


class MilvusCollection(object):
    def __init__(self, collection_name, dim, drop=False):
        """
        向量数据库集合
        :param collection_name:集合名称
        :param dim: 数据特征维度(Embedding-Dim)
        :param drop: True,如果collection存在,则删除;
        """
        self.drop = drop
        self.collection_name = collection_name
        self.collection = self.create(collection_name=self.collection_name, dim=dim, drop=self.drop)

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
    def create(collection_name, dim, drop=False) -> Collection:
        """
        :param collection_name: 集合名称
        :param dim:  数据特征维度(Embedding-Dim)
        :param drop: True,如果collection存在,则删除;
        :return:
        """
        if drop and utility.has_collection(collection_name):  # 删除原始的集合
            utility.drop_collection(collection_name=collection_name)
        fields = [
            FieldSchema(name='id', dtype=DataType.INT64, descrition='id of the embedding', is_primary=True,
                        auto_id=True),
            FieldSchema(name='info', dtype=DataType.JSON, descrition='data information', max_length=500),
            FieldSchema(name='feature', dtype=DataType.FLOAT_VECTOR, descrition='embedding vectors', dim=dim)
        ]
        schema = CollectionSchema(fields=fields, description='video dedup')
        collection = Collection(name=collection_name, schema=schema)
        return collection

    def insert(self, inputs: List):
        """
        :param inputs: [field0,field1,field2,...]
        :return:
        """
        self.collection.insert(inputs)  # 插入数据

    def search(self, vectors: List, metric_type="IP", top_k=3):
        """
        :param vectors: 查询向量 [v0,v1,v2]
        :param metric_type: L2，IP(必须归一化)
                          参考：https://milvus.io/docs/v2.2.x/metric.md?tab=floating
        :param top_k:
        :return:
        """
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
        search_params = {"metric_type": metric_type, "params": {"nprobe": 10}}
        # 执行搜索
        results = self.collection.search(data=vectors,
                                         anns_field="feature",
                                         output_fields=["id", "info", "feature"],
                                         param=search_params,
                                         limit=top_k,
                                         expr=None
                                         )
        # 返回结果
        outs = []
        for hits in results:
            r = [dict(id=hit.id, score=hit.score, fields=hit.fields) for hit in hits]
            outs.append(r)
        return outs

    def print_results(self, results: List):
        print("----" * 20)
        for i, res in enumerate(results):
            for info in res:
                print(f"i={i:4d}  {info}")
            print("----" * 20)


if __name__ == '__main__':
    pass
