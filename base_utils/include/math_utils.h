//
// Created by Pan on 2021/3/8.
//

#ifndef BASE_UTILS_MATH_UTILS_H
#define BASE_UTILS_MATH_UTILS_H

#include <vector>
#include <algorithm>
#include "opencv2/opencv.hpp"

#define PI 3.141592653589793
using namespace std;


/***
 * SoftMaX函数
 * @param src 输入
 * @param dst 输出
 * @param max_index 输出SoftMaX最大值的index
 * @param max_value 输出SoftMaX最大值，max_value = dst[max_index]
 */
void softmax(vector<float> src, vector<float> &dst, int &max_index, float &max_value);


inline float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}


/***
 * Sigmoid函数
 * @param x
 * @return
 */
inline float sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}


/***
 * 计算最大最小值
 * @tparam T
 * @param v
 * @param min_index
 * @param max_index
 */
template<typename T>
void find_min_max_indices(const std::vector<T> &v, int &min_index, int &max_index) {
    if (v.empty()) {
        min_index = max_index = -1; // 如果vector为空，则下标设置为-1
        return;
    }
    auto min_it = std::min_element(v.begin(), v.end());
    auto max_it = std::max_element(v.begin(), v.end());
    min_index = std::distance(v.begin(), min_it);
    max_index = std::distance(v.begin(), max_it);
}


/***
 * 计算两个Rect的IOU
 * @param r1
 * @param r2
 * @return
 */
float cv_iou(const cv::Rect &r1, const cv::Rect &r2);

/***
 * 计算两个Rect的IOU
 * @param r1
 * @param r2
 * @return
 */
float cv_iou2(const cv::Rect &r1, const cv::Rect &r2);


/***
 * 产生vector：P12 = point2-point1
 * @param point1
 * @param point2
 * @return
 */
cv::Point2f create_vector(cv::Point2f point1, cv::Point2f point2);

/***
 * 计算两个vector的夹角(0,180)
 * @param v1 vector1
 * @param v2 vector2
 * @param minangle：false,true：转为最小角度(0,90)
 * @return
 */
float vector_angle(cv::Point2f v1, cv::Point2f v2, bool minangle = false);


/***
 * 计算两个向量的乘积
 * @param v1
 * @param v2
 * @return
 */
float vector_multiply(vector<float> v1, vector<float> v2);

/***
 * 将弧度转换为角度
 * @param radian 弧度
 * @return
 */
float radian2angle(float radian);

/***
 * 计算两个点的欧式距离
 * @param v1
 * @param v2
 * @return
 */
float cal_distance(cv::Point2f p1, cv::Point2f p2);


#endif //BASE_UTILS_MATH_UTILS_H
