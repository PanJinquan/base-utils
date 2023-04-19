//
// Created by 390737991@qq.com on 2020/6/3.
//


#ifndef DETECTOR_POSE_FILTER_H
#define DETECTOR_POSE_FILTER_H

#include <vector>
#include "Types1.h"
#include <opencv2/opencv.hpp>
#include "filter/mean_filter.h"
#include "filter/kalman_filter.h"

#define FILTER MovingMeanFilter
//#define FILTER KalmanFilter

class PoseFilter {
private:
    vector<FILTER *> mFilters;
    vector<int> filter_id;
public:
    /***
     * 初始化Pose跟踪器(滤波器)
     * @param filter_id 需要跟踪的关键点ID，可指定多个
     * @param win_size 滑动窗口，用于记录历史信息的窗口大小，默认5
     * @param decay 权重衰减系数，值越大，历史影响衰减的越快，平滑力度越小,默认0.6
     */
    PoseFilter(vector<int> filter_id, int win_size = 5, float decay = 0.6);


    /***
     * 析构函数
     */
    ~PoseFilter();

    /***
     * 关键点跟踪(滤波)
     * @param info
     */
    void filter(ObjectInfo &info);
};

#endif //DETECTOR_POSE_FILTER_H
