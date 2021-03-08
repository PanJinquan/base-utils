//
// Created by dm on 2021/3/8.
//

#ifndef BASE_UTILS_MATH_UTILS_H
#define BASE_UTILS_MATH_UTILS_H

#include <vector>
#include <algorithm>

using namespace std;

/***
 * SoftMaX函数
 * @param src 输入
 * @param dst 输出
 * @param max_index 输出SoftMaX最大值的index
 * @param max_value 输出SoftMaX最大值，max_value = dst[max_index]
 */
void softmax(vector<float> &src, vector<float> &dst, int &max_index, float &max_value);


#endif //BASE_UTILS_MATH_UTILS_H
