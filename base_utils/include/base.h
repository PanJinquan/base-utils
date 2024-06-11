//
// Created by Pan on 2021/1/27.
//

#ifndef DETECTOR_BASE_H
#define DETECTOR_BASE_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

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
