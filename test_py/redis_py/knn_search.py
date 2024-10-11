# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2024-10-09 19:55:10
    @Brief  :
"""
from test_py.redis_py.redis_client import r_client, CacheMethod
from redisearch import Client, Query

if __name__ == "__main__":
    CacheMethod.set("name", "test")
    keys = CacheMethod().get_all_keys()
    print(keys)
