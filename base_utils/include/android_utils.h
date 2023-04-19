//
// Created by Pan on 2021/1/25.
//

#ifndef BASE_UTILS_ANDROID_UTILS_H
#define BASE_UTILS_ANDROID_UTILS_H

//#ifdef PLATFORM_LINUX
#ifdef PLATFORM_ANDROID
#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include "opencv2/opencv.hpp"

#define ASSERT_TRUE(status, ret)     if (!(status)) { return ret; }
#define ASSERT_FALSE(status)    ASSERT_TRUE(status, false)

/***
 * 将Android Bitmap转OpenCV Mat
 * @param env
 * @param obj_bitmap
 * @param matrix
 * @return
 */
bool BitmapToMatrix(JNIEnv *env, jobject obj_bitmap, cv::Mat &matrix) {
    void *bitmapPixels;         // Save picture pixel data
    AndroidBitmapInfo bitmapInfo; // Save picture parameters

    // Get picture parameters
    ASSERT_FALSE(AndroidBitmap_getInfo(env, obj_bitmap, &bitmapInfo) >= 0);
    // Only ARGB? 8888 and RGB? 565 are supported
    ASSERT_FALSE(bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGBA_8888
                 || bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGB_565);
    // Get picture pixels (lock memory block)
    ASSERT_FALSE(AndroidBitmap_lockPixels(env, obj_bitmap, &bitmapPixels) >= 0);
    ASSERT_FALSE(bitmapPixels);

    if (bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        // Establish temporary mat
        cv::Mat tmp(bitmapInfo.height, bitmapInfo.width, CV_8UC4, bitmapPixels);
        tmp.copyTo(matrix);// Copy to target matrix
    } else {
        cv::Mat tmp(bitmapInfo.height, bitmapInfo.width, CV_8UC2, bitmapPixels);
        cv::cvtColor(tmp, matrix, cv::COLOR_BGR5652RGB);
    }

    //convert RGB to BGR
    cv::cvtColor(matrix, matrix, cv::COLOR_RGB2BGR);
    //cv::cvtColor(matrix, matrix, cv::COLOR_RGB2BGRA);
    AndroidBitmap_unlockPixels(env, obj_bitmap);            // Unlock
    return true;
}


/***
 * 将OpenCV Mat转Android Bitmap
 * @param env
 * @param matrix
 * @param bitmap :  jobject bitmap,
 * @return
 */
bool MatrixToBitmap(JNIEnv *env, cv::Mat &matrix, jobject bitmap) {
    void *bitmapPixels;     // Save picture pixel data
    AndroidBitmapInfo bitmapInfo; // Save picture parameters
    // Get picture parameters
    ASSERT_FALSE(AndroidBitmap_getInfo(env, bitmap, &bitmapInfo) >= 0);
    // Only ARGB? 8888 and RGB? 565 are supported
    ASSERT_FALSE(bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGBA_8888
                 || bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGB_565);
    // It must be a 2-dimensional matrix with the same length and width
    ASSERT_FALSE(matrix.dims == 2 && bitmapInfo.height == (uint32_t) matrix.rows
                 && bitmapInfo.width == (uint32_t) matrix.cols);
    ASSERT_FALSE(matrix.type() == CV_8UC1 || matrix.type() == CV_8UC3 || matrix.type() == CV_8UC4);
    // Get picture pixels (lock memory block)
    ASSERT_FALSE(AndroidBitmap_lockPixels(env, bitmap, &bitmapPixels) >= 0);
    ASSERT_FALSE(bitmapPixels);

    if (bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        cv::Mat tmp(bitmapInfo.height, bitmapInfo.width, CV_8UC4, bitmapPixels);
        switch (matrix.type()) {
            case CV_8UC1:
                cv::cvtColor(matrix, tmp, cv::COLOR_GRAY2RGBA);
                break;
            case CV_8UC3:
                cv::cvtColor(matrix, tmp, cv::COLOR_RGB2RGBA);
                break;
            case CV_8UC4:
                matrix.copyTo(tmp);
                break;
            default:
                AndroidBitmap_unlockPixels(env, bitmap);
                return false;
        }
    } else {
        cv::Mat tmp(bitmapInfo.height, bitmapInfo.width, CV_8UC2, bitmapPixels);
        switch (matrix.type()) {
            case CV_8UC1:
                cv::cvtColor(matrix, tmp, cv::COLOR_GRAY2BGR565);
                break;
            case CV_8UC3:
                cv::cvtColor(matrix, tmp, cv::COLOR_RGB2BGR565);
                break;
            case CV_8UC4:
                cv::cvtColor(matrix, tmp, cv::COLOR_RGBA2BGR565);
                break;
            default:
                AndroidBitmap_unlockPixels(env, bitmap);
                return false;
        }
    }
    AndroidBitmap_unlockPixels(env, bitmap); // Unlock
    return true;
}


#endif
#endif //BASE_UTILS_ANDROID_UTILS_H
