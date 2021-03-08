//
// Created by dm on 2021/3/8.
//

#include "math_utils.h"
#include<cmath>

using namespace std;

void softmax(vector<float> &src, vector<float> &dst, int &max_index, float &max_value) {
    int length = src.size();
    max_index = max_element(src.begin(), src.end()) - src.begin();
    max_value = src[max_index];
    float denominator{0};
    for (int i = 0; i < length; ++i) {
        dst.push_back(std::exp(src[i] - max_value));
        denominator += dst[i];
    }
    for (int i = 0; i < length; ++i) {
        dst[i] /= denominator;
    }
    max_value = dst[max_index];
}


float cv_iou(const cv::Rect &r1, const cv::Rect &r2) {
    cv::Rect I = r1 | r2;//并集
    cv::Rect U = r1 & r2;//交集
    return U.area() * 1.f / I.area();
}

float cv_iou2(const cv::Rect &r1, const cv::Rect &r2) {
    // 计算每个矩形的面积
    int s1 = r1.width * r1.height;
    int s2 = r2.width * r2.height;
    // 计算相交矩形的面积
    int xmin = max(r1.x, r2.x);
    int ymin = max(r1.y, r2.y);
    int xmax = min(r1.x + r1.width, r2.x + r2.width);
    int ymax = min(r1.y + r1.height, r2.y + r2.height);
    int w = max(0, xmax - xmin);
    int h = max(0, ymax - ymin);
    float area = w * h;
    return area / (s1 + s2 - area);
}
