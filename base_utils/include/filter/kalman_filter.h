//
// Created by Pan on 2021/1/19.
//

#ifndef BASE_UTILS_KALMAN_FILTER_H
#define BASE_UTILS_KALMAN_FILTER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>

using namespace std;


/***
 * 卡尔曼滤波，可以有效解决关键点的抖动问题
 * 卡尔曼滤波模型假设k时刻的真实状态是从(k − 1)时刻的状态演化而来，符合下式：
 *      X(k) = F(k) * X(k-1) + B(k)*U(k) + W（k）
 * 其中：
 *      F(k)  是作用在xk−1上的状态变换模型（/矩阵/矢量）。
 *      B(k)  是作用在控制器向量uk上的输入－控制模型。
 *      W(k)  是过程噪声，并假定其符合均值为零，协方差矩阵为Qk的多元正态分布
 */
class KalmanFilter {
public:
    cv::KalmanFilter *mKF;
private:
    cv::Mat mMeasurement;

public:
    /***
     *
     * @param stateNum 状态值4×1向量(x,y,△x,△y)，坐标及速度
     * @param measureNum 测量值2×1向量(x,y)
     * @param controlParams
     */
    KalmanFilter(int stateNum = 4, int measureNum = 2);


    /***
     * 析构函数
     */
    ~KalmanFilter();

    /***
     * 更新数据
     * @param pos 测量值坐标点
     */
    void update(cv::Point pos);


    /***
     * 获得预测结果
     * @return
     */
    cv::Point predict();

};


#endif //BASE_UTILS_KALMAN_FILTER_H
