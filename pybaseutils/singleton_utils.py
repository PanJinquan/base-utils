# -*- coding:utf-8 -*-
"""
提供单例实现元类
"""
daemons_map = {}

class Singleton(type):
    """Singleton Metaclass"""
    
    def __init__(cls, name, bases, dic):
        super(Singleton, cls).__init__(name, bases, dic)
        cls.instance = None
        
    def __call__(cls, *args, **kwargs):
        name = cls.__name__
        ob = None
        if name not in daemons_map:
            ob = super(Singleton, cls).__call__(*args, **kwargs)
            daemons_map[name] = ob
        else:
            ob = daemons_map[name]
        return ob
