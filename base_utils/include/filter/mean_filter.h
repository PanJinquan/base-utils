//
// Created by Pan on 2021/1/20.
//

#ifndef BASE_UTILS_MEAN_FILTER_H
#define BASE_UTILS_MEAN_FILTER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>

using namespace std;

/***
 * 加权平均滑动滤波法(Weighted Moving Average Filter)，
 * 可以有效解决关键点的抖动问题
 */
class MovingMeanFilter {
public:
    vector<cv::Point> mQueue;
private:
    int mWinSize;
    cv::Mat mWeightDecay;
public:

    /***
     * 构造函数
     * @param win_size 滑动窗口，用于记录历史信息的窗口大小，默认5
     * @param decay 权重衰减系数，值越大，历史影响衰减的越快，平滑力度越小,默认0.6
     */
    MovingMeanFilter(int win_size = 5, float decay = 0.6);

    /***
     * 析构函数
     */
    ~MovingMeanFilter();

    /***
     * 更新数据
     * @param pos 坐标点
     */
    void update(cv::Point pos);


    /***
     * 获得预测结果
     * @return 返回预测的坐标值
     */
    cv::Point predict();

private:
    /***
     * 实现加权平均平滑，权重由weight_decay确定
     * @return
     */
    cv::Point filter();

    /***
     * 获得权重衰减
     * 当n=5,decay=0.5时，对应的衰减权重为，越远的权重越小
     * w=[0.0625 0.0625 0.125  0.25   0.5   ]
     * @param n
     * @param decay 衰减系数，值越大，历史影响衰减的越快，平滑力度越小
     * @return
     */
    vector<float> get_weight_decay(int n, float decay);
};

#endif //BASE_UTILS_MEAN_FILTER_H
