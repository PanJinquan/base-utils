# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2024-10-29 11:33:21
    @Brief  : https://blog.csdn.net/jixiaoyu0209/article/details/140444906
              https://www.modb.pro/db/578430
"""
import random
import numpy as np
from typing import List
from test_py.milvus_demo import milvus_client
from pybaseutils import numpy_utils

collection_name = "example01"  # 集合名称
DATA_DIMS = 5  # 数据特征维度(Embedding-Dim)

collection = milvus_client.MilvusCollection(collection_name=collection_name, dim=DATA_DIMS, drop=True)

collection.get_collections()
# collection.del_collections(keys=None)

def example01():
    data_nums = 10  # 样本个数
    feature = [[i] * DATA_DIMS for i in range(data_nums)]
    feature = np.asarray(feature, dtype=np.float32)  # (nums,dim)
    feature[:, -1] = 0.1
    feature = numpy_utils.feature_norm(feature)  # 特征归一化
    feature = feature.tolist()
    info = [{"file": f"image_{i:0=3}.jpg", "box": [], "name": "babel"} for i in range(10)]
    # TODO collection.insert([info, feature])
    for d, f in zip(info, feature): collection.insert([[d], [f]])
    # collection.insert([info, feature])
    results = collection.search([feature[5], feature[2]], top_k=2)
    collection.print_results(results)


if __name__ == '__main__':
    example01()
