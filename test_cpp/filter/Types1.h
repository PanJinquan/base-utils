//
// Created by 390737991@qq.com on 2020/6/3.
//


#ifndef BODY_DETECTION_RK3399_TYPES_H
#define BODY_DETECTION_RK3399_TYPES_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "debug.h"
#include "Interpreter.h"

using namespace std;

/***
 * 关键点(包含一个坐标点point和分数score)
 */
struct KeyPoint {
    float score;//关键点分数
    cv::Point2f point;
};

/***
 * 目标信息(包含目标的多个关键点keypoints和检测区域框rect)
 */
struct ObjectInfo {
    float x1, x2, y1, y2, score;   // 左上角x1,左上角y1,右下角x2,右下角y2,置信度分数score
    int label;                     // 检测框的类别ID，即bboxes框的index,查询ModelParam的class_names可获得真实的类别
    vector<KeyPoint> keypoints;
    vector<cv::Point2f> landmarks; // 人脸关键点，如果ModelParam的num_landmarks为0，则不存在关键点信息

};

/***
 * 帧信息(帧图像中多个目标的信息)
 */
struct FrameInfo {
    vector<ObjectInfo> info;
};


#endif //BODY_DETECTION_RK3399_TYPES_H
