# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2023-09-07 09:10:14
    @Brief  :
"""
from typing import Tuple, List, Dict, Callable
from pybaseutils.singleton_utils import Singleton


class Register(metaclass=Singleton):

    def __init__(self, name):
        self.module_dict = dict()
        self.name = name

    def put_module(self, func=None, name=""):
        def decorator(task: Callable):
            name_ = name if name else task.__name__
            print(name_)
            # assert name_ not in self.module_dict, Exception(name_)
            self.module_dict[name_] = task
            return task

        if func:
            name_ = name if name else func.__name__
            # assert name_ not in self.module_dict, Exception(name_)
            self.module_dict[name_] = func.__func__  # 将类方法转为函数
            return func
        else:
            return decorator

    def get_module(self, name):
        return self.module_dict[name]

    def modules(self):
        return dict(self.module_dict)


register = Register("Component")
