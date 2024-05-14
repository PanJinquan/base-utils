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
 * @tparam S
 * @tparam T
 * @param src
 * @return
 */
template<class S, class T>
vector<T> vector_type(vector<S> src) {
    vector<T> dst;
    // 使用std::transform转换类型
    std::transform(src.begin(), src.end(), dst.begin(),
                   [](S value) { return T(value); });
    return dst;
}


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


#endif //DETECTOR_BASE_H
