# -*- coding:utf-8 -*-
"""
提供模型相关方法
"""
import logging
import threading
import time
import asyncio
from pybaseutils import json_utils


class EvaluationMethod():
    """
    Evaluation核心实现方法类
    """

    def __init__(self):
        self.msg = "ok"

    async def calligraphy_hard_evaluation(self, params):
        results = {"result": params, "msg": self.msg}
        return results


if __name__ == '__main__':
    w = EvaluationMethod()
    params = {"data": []}
    # 使用 asyncio.run() 来运行异步方法
    result = asyncio.run(w.calligraphy_hard_evaluation(params))
    print(result)