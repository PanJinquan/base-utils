# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2024-10-29 11:33:21
    @Brief  : https://blog.csdn.net/jixiaoyu0209/article/details/140444906
"""
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# 连接 Milvus 服务
connections.connect(host='localhost', port='19530')

# 指定集合名和字段
collection_name = 'test_collection'
vector_field_name = 'vec_field'

# 创建集合（如果已存在，则不需要这一步）
dim = 128
field = FieldSchema(name=vector_field_name, dtype=DataType.FLOAT_VECTOR, is_primary=True, dim=dim)
schema = CollectionSchema(fields=[field], description="test collection")

# 如果集合不存在则创建
if collection_name not in collections.list_collections():
    collection = Collection(name=collection_name, schema=schema)

# 查询当前集合
collection = Collection(name=collection_name)
collection.load()  # 加载集合

# 查询参数
search_param = {
    "metric_type": "L2",
    "params": {
        "nprobe": 10
    }
}

# 查询向量
query_vector = [0.1, 0.2] * dim  # 假设的查询向量
search_results = collection.search(vector_field_name=vector_field_name, query_records=[query_vector],
                                   top_k=10, params=search_param)

# 打印搜索结果
print(search_results)