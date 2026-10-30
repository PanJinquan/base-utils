# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2025-07-08 14:10:15
# @Brief  : 转换labelme标注数据为voc格式
# --------------------------------------------------------
"""

import queue

q = queue.Queue(maxsize=10)
q.put(1)
q.put(2)
print(q.qsize())
q.queue.clear()
print(q.qsize())

