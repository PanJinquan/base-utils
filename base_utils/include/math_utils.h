//
// Created by dm on 2021/3/8.
//

#ifndef BASE_UTILS_MATH_UTILS_H
#define BASE_UTILS_MATH_UTILS_H

#include <vector>
#include <algorithm>
#include "opencv2/opencv.hpp"

using namespace std;

/***
 * SoftMaX函数
 * @param src 输入
 * @param dst 输出
 * @param max_index 输出SoftMaX最大值的index
 * @param max_value 输出SoftMaX最大值，max_value = dst[max_index]
 */
void softmax(vector<float> &src, vector<float> &dst, int &max_index, float &max_value);


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

#endif //BASE_UTILS_MATH_UTILS_H
