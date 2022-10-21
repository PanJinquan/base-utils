#include <iostream>
#include <omp.h>   // NEW ADD
#include<opencv2/opencv.hpp>
#include<string>
#include "debug.h"
#include "image_utils.h"

using namespace std;

#define ITERA_NUMS 50
#define THREAD_NUMS 4

void test1(int iter_nums) {
#pragma omp parallel for num_threads(THREAD_NUMS)// NEW ADD
    for (int i = 0; i < iter_nums; i++) {
        LOGI("i=%d", i);
    }
}

void test2(int iter_nums) {
    string path = "../../data/test_image/test1.jpg";
    cv::Mat src = cv::imread(path);
    cv::Mat dst;
    int resize_width = 1000;
    int resize_height = 1000;
    DEBUG_TIME(t1);
    for (int i = 0; i < iter_nums; i++) {
        LOGI("i=%d", i);
        cv::resize(src, dst, cv::Size(resize_width, resize_height));
        //dst = image_resize(src, resize_width, resize_height);
    }
    DEBUG_TIME(t2);
    DEBUG_TIME(t3);
    //#pragma omp parallel for num_threads(THREAD_NUMS) private(dst)
#pragma omp parallel for num_threads(THREAD_NUMS)
    for (int i = 0; i < iter_nums; i++) {
        LOGI("i=%d", i);
        cv::resize(src, dst, cv::Size(resize_width, resize_height));
        //dst = image_resize(src, resize_width, resize_height);
    }
    DEBUG_TIME(t4);
    LOGI("Single-Thread:%3.3fms", RUN_TIME(t2 - t1));
    LOGI("Multi-Thread :%3.3fms", RUN_TIME(t4 - t3));

}


int main() {
#if _OPENMP
    LOGI("support openmp");
#else
    LOGI("not support openmp");
#endif
    //test1(ITERA_NUMS);
    test2(ITERA_NUMS);
    return 0;
}