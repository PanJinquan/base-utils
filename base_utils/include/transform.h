//
// Created by Pan on 2024/4/24.
//

#ifndef DETECTOR_TRANSFORM_H
#define DETECTOR_TRANSFORM_H

#include <vector>
#include <string>
#include "math_utils.h"
#include "transform.h"
#include "image_utils.h"


/***
 * 按顺时针方向进行排序:[top-left, top-right, bottom-right, bottom-left]
 * top-left    ：对应y+x之和的最小点
 * bottom-right：对应y+x之和的最大点
 * top-right   ：对应y-x之差的最小点
 * bottom-left ：对应y-x之差的最大点
 * point is-->
 *     0(top-left)----(w10)----1(top-right)
 *        |                       |
 *      (h30)                    (h21)
 *        |                       |
 *    3(bottom-left)--(w23)---2(bottom-right)
 * @param inp 输入至少4个点
 * @param dst 返回4个点，按顺时针方向进行排序:[top-left, top-right, bottom-right, bottom-left]
 */
void get_order_points(vector<cv::Point2f> inp, vector<cv::Point2f> &dst);


/***
 * 获得旋转矩形框，即最小外接矩形的四个角点
 * @param src_pts
 * @param dst_pts
 * @param order 对4个点按顺时针方向进行排序:[top-left, top-right, bottom-right, bottom-left]
 */
void get_obb_points(vector<cv::Point2f> src_pts, vector<cv::Point2f> &dst_pts, bool order = true);

void get_obb_points(vector<cv::Point> src_pts, vector<cv::Point2f> &dst_pts, bool order = true);

/***
 * 根据输入的四个角点，计算其矫正后的目标四个角点,src_pts四个点分布：
 * @param src_pts 输入四个角点，必须是有序的，[top-left, top-right, bottom-right, bottom-left]
 * @param dst_pts 输出矫正后的目标四个角点；如果要获得矫正后的长宽，则使用dst_pts[2]
 */
void get_target_points(vector<cv::Point2f> src_pts, vector<cv::Point2f> &dst_pts);

/***
 * 计算变换矩阵
 * @param src_pts 输入原始关键点
 * @param dst_pts 输入目标关键点
 * @param M 输出变换矩阵
 * @param method 使用方法 0-使用estimateAffine2D计算变换矩阵 ;1-使用最小二乘法计算变换矩阵
 */
void get_transform(vector<cv::Point2f> &src_pts, vector<cv::Point2f> &dst_pts, cv::Mat &M, int method = 0);

/***
 * 计算变换矩阵
 * @param src_pts 输入原始关键点
 * @param dst_pts 输入目标关键点
 * @param M 输出变换矩阵
 * @param Minv 输出逆变换矩阵
 * @param method 使用方法 0-使用estimateAffine2D计算变换矩阵 ;1-使用最小二乘法计算变换矩阵
 */
void get_transform(vector<cv::Point2f> &src_pts, vector<cv::Point2f> &dst_pts,
                   cv::Mat &M, cv::Mat &Minv, int method = 0);


/***
 * 通过最小二乘法计算变换矩阵
 * @param src_pts 输入原始关键点
 * @param dst_pts 输入目标关键点
 * @return
 */
cv::Mat solve_lstsq(vector<cv::Point2f> &src_pts, vector<cv::Point2f> &dst_pts);


/***
 * 对图像进行仿生变换
 * @param image input image
 * @param src_pts 原始点
 * @param dst_pts 目标点，当dst_pts为空时，则基于src_pts计算矫正后目标点
 * @param dsize 返回图像的大小
 * @param scale 返回图像缩放大小
 * @param flags
 * @param borderMode
 * @param color is borderValue
 * @return
 */
cv::Mat image_alignment(cv::Mat &image,
                        vector<cv::Point2f> src_pts,
                        vector<cv::Point2f> &dst_pts,
                        cv::Size dsize = cv::Size(-1, -1),
                        cv::Size2f scale = cv::Size(1.0, 1.0),
                        int flags = cv::INTER_LINEAR,
                        int borderMode = cv::BORDER_CONSTANT,
                        cv::Scalar color = COLOR_BLACK);


#endif //DETECTOR_TRANSFORM_H
