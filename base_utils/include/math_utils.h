//
// Created by dm on 2021/3/8.
//

#ifndef BASE_UTILS_MATH_UTILS_H
#define BASE_UTILS_MATH_UTILS_H


template<class TYPE>
static void softmax(vector <TYPE> &src, vector <TYPE> &dst, int &max_index, TYPE &max_value) {
    int length = src.size();
    max_index = max_element(src.begin(), src.end()) - src.begin();
    max_value = src[max_index];
    TYPE denominator{0};
    for (int i = 0; i < length; ++i) {
        dst.push_back(std::exp(src[i] - max_value));
        denominator += dst[i];
    }
    for (int i = 0; i < length; ++i) {
        dst[i] /= denominator;
    }
    max_value = dst[max_index];
}


#endif //BASE_UTILS_MATH_UTILS_H
