# -*- coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2023-09-07 09:11:05
    @Brief  :
"""

from pybaseutils.singleton_utils import Singleton


class Plugin(metaclass=Singleton):
    def __init__(self):
        self.name = None


if __name__ == "__main__":
    p1 = Plugin()
    p1.name = "p1"
    p2 = Plugin()
    p2.name = "p2"
    print(p1, p1.name)
    print(p2, p2.name)
