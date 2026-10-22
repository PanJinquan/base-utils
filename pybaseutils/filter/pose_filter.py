# -*- coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2021-07-23 09:51:36
"""
from pybaseutils.filter import mean_filter, kalman_filter, motion_filter


class PoseFilter(object):
    def __init__(self, kid, wsize=10, decay=0.4):
        """
        :param kid: 需要跟踪的关键点ID，可指定多个,如[0,1,2,3]
        :param wsize: 滑动窗口，用于记录历史信息的窗口大小，默认5
        :param decay: 权重衰减系数，值越大，历史影响衰减的越快，平滑力度越小,默认0.6
        """
        self.kid = kid
        self.filters = []
        for i in range(len(kid)):
            filter = mean_filter.MeanFilter(wsize=wsize, decay=decay)
            # filter = kalman_filter.KalmanFilter(stateNum=4, measureNum=2)
            self.filters.append(filter)

    def filter(self, points):
        for i in range(len(self.filters)):
            id = self.kid[i]
            self.filters[i].update(points[id, :].copy())
            points[id, :] = self.filters[i].predict()
