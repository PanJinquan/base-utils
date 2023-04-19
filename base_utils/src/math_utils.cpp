//
// Created by Pan on 2021/3/8.
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

cv::Point2f create_vector(cv::Point2f point1, cv::Point2f point2) {
    // P12 = point2-point1
    return point2 - point1;
}

float vector_angle(cv::Point2f v1, cv::Point2f v2, bool minangle) {
    // cosφ = u·v/|u||v|
    float lx = sqrt(v1.dot(v1));
    float ly = sqrt(v2.dot(v2));
    float value = v1.dot(v2) / ((lx * ly) + 1e-6);
    float radian = acos(value);
    float angle = radian2angle(radian);
    if (minangle) {
        angle = angle < 90 ? angle : 180 - angle;
    }
    return angle;
}

float vector_multiply(vector<float> v1, vector<float> v2) {
    int size = v1.size();
    float angle = 0.f;
    for (int i = 0; i < size; ++i) {
        angle += v1[i] * v2[i];
    }
    return angle;
}

float radian2angle(float radian) {
    float angle = radian * (180 / PI);
    return angle;
}
