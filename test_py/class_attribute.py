# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-12-31 11:37:30
    @Brief  : https://blog.csdn.net/qdPython/article/details/121381363
"""
import types


class Base():
    def add(self, a, b):
        info = "Base.add: {}+{}={}".format(a, b, a + b)
        return info

    def register(self, name):
        c = compile(f"self.{name} = types.MethodType(self.fun2, self)", '', 'exec')
        # c = compile(f"self.{name} = types.FunctionType(self.fun2, self)", '', 'exec')
        # eval(c)
        self.fun3 = types.MethodType(self.fun2, self)
        fun = getattr(self, "fun3")
        print("动态获得类方法: register", fun, fun(10))

    @classmethod
    def fun2(cls, a, b):
        info = "fun22: {}+{}={}".format(a, b, a + b)
        return info


def fun2(self, a, b):
    info = "fun21: {}+{}={}".format(a, b, a + b)
    return info


if __name__ == '__main__':
    b = Base()
    print("动态获得类方法:", getattr(b, "add")(1, 2))

    # b.fun2 = types.MethodType(fun2, b)
    # print("动态添加类方法:", b.fun2(3, 4))

    name = "fun3"
    # c = compile(f"b.{name} = types.MethodType(fun2, b)", '', 'exec')
    # eval(c)
    # print("动态获得类方法:", name, getattr(b, name)(10, 20))

    b.register(name=name)
