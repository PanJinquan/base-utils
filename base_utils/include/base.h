//
// Created by Pan on 2021/1/27.
//

#ifndef DETECTOR_BASE_H
#define DETECTOR_BASE_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace std;

/***
 * 单个跟踪目标的信息
 */
struct Track {
    int idx = -1;                       // 跟踪目标ID（唯一且连续递增）
    int lost = 0;                       // 连续丢失检测的帧数
    int index = -1;                     // 当前帧匹配的检测结果索引(未匹配为-1)
    float distance = 0.0f;              // 最近一次的测距结果(米)
    float speed = 0.0f;                 // 最近一次的速度结果(km/h)
    vector<cv::Point2f> trajectory;     // 目标中心点的运动轨迹
};


/***
 * 目标信息
 */
struct ImageInfo {
    float x1, x2, y1, y2, score;   // 目标：左上角x1,左上角y1,右下角x2,右下角y2,置信度分数score
    int label;                     // 目标类别ID
    Track track;                   // 跟踪目标的信息
};


/***
 * 释放vector空间
 * @tparam T
 * @param vt
 */
template<class T>
void clear_vector(std::vector<T> &vt) {
    vt.clear();
    std::vector<T> vtTemp;
    vtTemp.swap(vt);
    //vector<T>().swap(queue);
};


/***
 * 转换vector的数据类型
 * @tparam S_TYPE :原始数据类型
 * @tparam D_TYPE :目标数据类型
 * @param src
 * @return
 */
template<class S_TYPE, class D_TYPE>
vector<D_TYPE> vector_type(vector<S_TYPE> src) {
    // 使用std::transform转换类型
    std::vector<D_TYPE> dst(src.size());
    std::transform(src.begin(), src.end(), dst.begin(),
                   [](S_TYPE value) { return D_TYPE(value); });
    return dst;
};


/***
 * 将数组array转为vector形式
 * @param pts
 * @return
 */
template<class T>
vector<T> array2vector(T *pts, int num) {
    vector<T> out;
    for (int i = 0; i < num; ++i) {
        out.push_back(pts[i]);
    }
    return out;
};

/***
 * 将数组string转为vector形式
 * @param pts
 * @return
 */
inline vector<string> string2vector(string str) {
    vector<string> out;
    for (size_t i = 0; i < str.size(); ++i) {
        out.push_back(str.substr(i,1));
    }
    return out;
};

/**
 * 将数值转换为string
 * @param format
 * @param i
 * @return
 */
template<class T>
string tostring(string format, T i) {
    char t[256];
    sprintf(t, format.c_str(), i);
    string s(t);
    return s;
};


#endif //DETECTOR_BASE_H
