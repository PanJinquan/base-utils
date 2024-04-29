# -*- coding:utf-8 -*-
"""
    @Author : PKing
    @E-mail :
    @Date   :
    @Brief  : 单实例模式，多次创建返回的对象地址是一样，因此多次初始化实例，以第一次初始化的参数为参数
              如果通过对象修改属性变量，则以最后一次修改为结果
"""
import threading

daemons_map = {}


class SingletonType(type):
    """建议使用这个Singleton Metaclass"""

    def __init__(cls, name, bases, dic):
        super(SingletonType, cls).__init__(name, bases, dic)
        cls.instance = None

    def __call__(cls, *args, **kwargs):
        name = cls.__name__
        ob = None
        if name not in daemons_map:
            ob = super(SingletonType, cls).__call__(*args, **kwargs)
            daemons_map[name] = ob
        else:
            ob = daemons_map[name]
        return ob


def synchronized(func):
    func.__lock__ = threading.Lock()

    def lock_func(*args, **kwargs):
        with func.__lock__:
            return func(*args, **kwargs)

    return lock_func


class Singleton(type):
    def __init__(cls, *args, **kwargs):
        cls.__instance = None
        super().__init__(*args, **kwargs)

    @synchronized
    def __call__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = super().__call__(*args, **kwargs)
        return cls.__instance
