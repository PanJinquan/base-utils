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
