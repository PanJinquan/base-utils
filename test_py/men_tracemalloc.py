# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2023-02-24 16:53:01
    @Brief  : http://study.yali.edu.cn/pythonhelp/library/tracemalloc.html#module-tracemalloc
"""
import tracemalloc
tracemalloc.start()

snapshot0=tracemalloc.take_snapshot() # 第一张快照
d = [dict(zip('xy', (5, 6))) for i in range(1000000)]
t = [tuple(zip('xy', (5, 6))) for i in range(1000000)]
snapshot1 = tracemalloc.take_snapshot()  # 快照，当前内存分配
# top_stats = snapshot1.statistics('lineno') # 快照对象的统计
top_stats = snapshot1.compare_to(snapshot0, 'lineno')  # 快照对比

for stat in top_stats:
    print(stat)
