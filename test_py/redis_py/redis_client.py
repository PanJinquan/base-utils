# -*- coding: utf-8 -*-
"""
redis client
"""
import redis
from redis import StrictRedis

REDIS_URI = 'redis://@10.12.51.100:6379'
pool = redis.ConnectionPool().from_url(REDIS_URI)
r_client = StrictRedis(connection_pool=pool)


class CacheMethod:
    """redis method"""

    def __init__(self):
        pass

    @staticmethod
    def get_client():
        return r_client

    @staticmethod
    def set(key, value):
        return r_client.set(key, value)

    @staticmethod
    def get(key):
        return r_client.get(key)

    @staticmethod
    def exists(key):
        return r_client.exists(key)

    @staticmethod
    def delete(key):
        return r_client.delete(key)

    @staticmethod
    def keys(key):
        return r_client.keys(key)

    @staticmethod
    def incr(key):
        return r_client.incr(key)

    @staticmethod
    def decr(key):
        return r_client.decr(key)

    @staticmethod
    def expire(key, time):
        """设置，过期删除时间，必须创建后再设置"""
        return r_client.expire(key, time)

    @staticmethod
    def get_all_keys():
        """获得所有keys"""
        keys = r_client.keys('*')
        keys = keys if keys else []
        keys = [k.decode('utf-8') for k in keys]
        return keys

    @staticmethod
    def del_all_keys(ignore=[]):
        """删除所有keys"""
        keys = CacheMethod.get_all_keys()
        try:
            for k in keys:
                if k in ignore: continue
                CacheMethod.delete(k)
                print("INFO: Redis delete key:{}".format(k))
        except Exception as e:
            print(e)
            print("Error: Redis delete key:{}".format(k))

    @staticmethod
    def set_all_expire(time, ignore=[]):
        """设置所有keys的过期时间"""
        keys = CacheMethod.get_all_keys()
        try:
            for k in keys:
                if k in ignore: continue
                CacheMethod.expire(k, time=time)
                print("INFO: Redis set key={}, expire={}s".format(k, time))
        except Exception as e:
            print(e)
