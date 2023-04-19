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
void clear_vector(std::vector <T> &vt) {
    vt.clear();
    std::vector <T> vtTemp;
    vtTemp.swap(vt);
    //vector<T>().swap(queue);
};

#endif //DETECTOR_BASE_H
