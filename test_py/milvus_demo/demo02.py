# -*- coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2024-10-29 11:33:21
    @Brief  : https://blog.csdn.net/jixiaoyu0209/article/details/140444906
"""
import cv2
import numpy as np
from pymilvus import MilvusClient
from pymilvus import MilvusClient, DataType
from pybaseutils import image_utils, file_utils

client = MilvusClient("http://192.168.2.52:19530")
collection_name = "example04"  # 集合名称
dim = 10 * 10


def create():
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name="name", datatype=DataType.VARCHAR, max_length=256)
    schema.verify()
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="id",
        index_type="STL_SORT"
    )
    index_params.add_index(
        field_name="vector",
        index_type="IVF_FLAT",
        metric_type="L2",
        params={"nlist": 1024}
    )
    # 创建 collection
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )


def insert():
    image_dir = "/media/PKing/新加卷/SDK/base-utils/data/test_image"
    image_list = file_utils.get_files_list(image_dir)
    for image_file in image_list:
        print(image_file)
        image = image_utils.read_image(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image_utils.resize_image(image, size=(10, 10))
        # 提取特征向量
        features = np.asarray(image.reshape(-1) / 255.0, dtype=np.float32)
        # milvus
        client.insert(
            collection_name=collection_name,
            data={
                "vector": features,
                "name": image_file
            }
        )
    client.flush(collection_name=collection_name)


if __name__ == '__main__':
    create()
    insert()
