//
// Created by 390737991@qq.com on 2018/6/3.
//

#ifndef FACE_ALIGNMENT_H
#define FACE_ALIGNMENT_H

#include <vector>
#include <string>
#include "image_utils.h"

using namespace std;

class FaceAlignment {

public:
    /***
     * 初始化人脸校准：FaceAlignment
     * @param faceWidth: face width
     * @param faceHeight: face height
     * @param ex: 宽度扩展比例
     * @param ey: 长度扩展比例
     * @param square: true or false
     */
    explicit FaceAlignment(int faceWidth = 112, int faceHeight = 112, float ex = 1.f, float ey = 1.f,
                           bool square = true);

    /**
     * 析构函数
     */
    ~FaceAlignment();


    /***
     * 使用alignment的方法裁剪人脸
     * @param image 输入原始图像
     * @param landmarks 输入原始关键点
     * @param out_face 输出校准的人脸图像
     */
    void crop_faces_alignment(cv::Mat &image, vector<cv::Point2f> &landmarks, cv::Mat &out_face);

    /***
     * 使用alignment的方法裁剪人脸
     * @param image 输入原始图像
     * @param landmarks 输入原始关键点
     * @param out_face 输出校准的人脸图像
     * @param M 输出变换矩阵
     * @param Minv 输出逆变换矩阵
     */
    void crop_faces_alignment(cv::Mat &image, vector<cv::Point2f> &landmarks,
                              cv::Mat &out_face, cv::Mat &M, cv::Mat &Minv);

    /***
     * 不使用alignment的方法裁剪人脸
     * @param image 输入原始图像
     * @param rect
     * @param out_face 输出人脸图像
     */
    void crop_faces(cv::Mat &image, cv::Rect &rect, cv::Mat &out_face);

private:
    int face_width, face_height;
    // landmark个数
    const int num_landmarks{5};
    // 标准人脸的大小(96×112)
    const int size_ref[2] = {96, 112};
    // 标准人脸(96×112)的5个关键点
    const cv::Point2f kpts_ref[5] = {
            {30.29459953, 51.69630051},
            {65.53179932, 51.50139999},
            {48.02519989, 71.73660278},
            {33.54930115, 92.3655014},
            {62.72990036, 92.20410156}
    };
    // 目前人脸关键点
    cv::Point2f *dst_pts;
};

#endif //FACE_ALIGNMENT_H
