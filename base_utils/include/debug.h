#ifndef DETECT_DEBUG_H
#define DETECT_DEBUG_H

#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"
#include "image_utils.h"
#include <chrono>
#include <assert.h>

using namespace std;

/***
 * 设置LOG的信息开关
 */
//debug info ON-OFF
//CMake setting add_definitions(-DDEBUG_ON)
#define DEBUG_OFF
#ifdef  DEBUG_ON
#define DEBUG_LOG_OFF         //Window debug:print debug info
#define DEBUG_IMSHOW_OFF      //show debug images
#define DEBUG_IMWRITE_OFF     //write debug images
#define DEBUG_ANDROID_OFF      //android debug on/off

#else
#define DEBUG_OFF(format, ...)
#endif

/***
 * 其他宏定义
 */
#define LOG_TAG    "cv-jni-log"
#define millisecond 1000000
#define DEBUG_TIME_ON         //run times test on/off
#define FILE_INFO printf("[%s line%d] [tag:%s] ",__FILE__,__LINE__,LOG_TAG);
#define ASSERT(...) assert( __VA_ARGS__)
#define CV_ASSERT(...) CV_Assert( __VA_ARGS__)


/***
 * 打印信息定义
 */
//print debug info
#ifdef DEBUG_ANDROID_ON
#include <android/log.h>
// Define the LOGI and others for print debug infomation like the log.i in java
#define LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG, __VA_ARGS__)
#define LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG, __VA_ARGS__)
#define LOGW(...)  __android_log_print(ANDROID_LOG_WARN,LOG_TAG, __VA_ARGS__)
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG, __VA_ARGS__)
#define LOGF(...)  __android_log_print(ANDROID_LOG_FATAL,LOG_TAG, __VA_ARGS__)
#define DEBUG_PRINT(format, ...)
#define DEBUG_COUT(...)
#elif defined DEBUG_LOG_ON
#define LOGI(...)  FILE_INFO;printf(__VA_ARGS__); printf("\n")
#define LOGD(...)  FILE_INFO;printf(__VA_ARGS__); printf("\n")
#define LOGW(...)  FILE_INFO;printf(__VA_ARGS__); printf("\n")
#define LOGE(...)  FILE_INFO;printf(__VA_ARGS__); printf("\n")
#define LOGF(...)  FILE_INFO;printf(__VA_ARGS__); printf("\n")
#define DEBUG_PRINT(...) printf( __VA_ARGS__);printf("\n")
#define DEBUG_COUT(...) std::cout __VA_ARGS__ << std::endl
#else
#define LOGI(...)
#define LOGD(...)
#define LOGW(...)
#define LOGE(...)
#define LOGF(...)
#define DEBUG_PRINT(format, ...)
#define DEBUG_COUT(...)
#endif

/***
 * run time define
 */
#ifdef  DEBUG_TIME_ON
//设置计算运行时间的宏定义
#define DEBUG_TIME(time_) auto time_ =std::chrono::high_resolution_clock::now()
#define RUN_TIME(time_)  (double)(time_).count()/millisecond
#else
#define DEBUG_TIME(time_)
#define RUN_TIME(time_)
#endif

/***
 * show debug images define
 */
#ifdef  DEBUG_IMSHOW_ON
#define DEBUG_IMSHOW(...) image_show(__VA_ARGS__)
#else
#define DEBUG_IMSHOW(format, ...)
#endif

/***
 * write debug images define
 */
#ifdef  DEBUG_IMWRITE_ON
#define DEBUG_IMWRITE(...) image_save(__VA_ARGS__)
#else
#define DEBUG_IMWRITE(format, ...)
#endif


template<typename TYPE>
void PRINT_1D(string name, TYPE *p1, int len) {
    LOGD("%s", name.c_str());
    for (int i = 0; i < len; i++) {
        LOGD("%f,", p1[i]);
    }
    cout << endl;
}

template<typename TYPE>
void PRINT_VECTOR(string tag, vector<TYPE> v) {
    LOGD("%s", tag.c_str());
    for (int i = 0; i < v.size(); ++i) {
        LOGD("i=%d,%f,", i, v[i]);
    }
    LOGD("\n");

};
#endif