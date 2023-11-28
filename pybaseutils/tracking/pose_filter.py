# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2021-07-23 09:51:36
"""
from pybaseutils.tracking import mean_filter, kalman_filter, motion_filter


class PoseFilter(object):
    def __init__(self, filter_id, win_size=10, decay=0.4):
        self.filter_id = filter_id
        self.filters = []
        for i in range(len(filter_id)):
            filter = motion_filter.MotionFilter(win_size=win_size, decay=0.6)
            # filter = mean_filter.MeanFilter(win_size=win_size, decay=decay)
            # filter = kalman_filter.KalmanFilter(stateNum=4, measureNum=2)
            self.filters.append(filter)

    def filter(self, points):
        for i in range(len(self.filters)):
            id = self.filter_id[i]
            self.filters[i].update(points[id, :].copy())
            points[id, :] = self.filters[i].predict()
