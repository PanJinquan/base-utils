//
// Created by dm on 2021/1/15.
//

#ifndef DETECTOR_IMAGE_UTILS_H
#define DETECTOR_IMAGE_UTILS_H

#include <string>
#include "opencv2/opencv.hpp"

using namespace std;


/***
 * 缩放图片，若resize_width或者resize_height，有一个是小于等于0，则进行等比例缩放图片
 * @param src
 * @param resize_width 默认-1
 * @param resize_height 默认-1
 * @return
 */
cv::Mat image_resize(cv::Mat &src, int resize_width = -1, int resize_height = -1);


/***
 * 图像裁剪,超出的区域会被丢弃
 * @param src
 * @param rect
 * @return
 */
cv::Mat image_crop(cv::Mat &src, cv::Rect rect);


/***
 *  图像裁剪,超出的区域会被丢弃
 * @param src
 * @param x1
 * @param x2
 * @param y1
 * @param y2
 * @return
 */
cv::Mat image_crop(cv::Mat &src, int x1, int x2, int y1, int y2);


/***
 * 图像裁剪,超出的区域会被填充
 * @param src
 * @param rect
 * @param color 填充的颜色，默认黑色 color = (0, 0, 0)
 * @return
 */
cv::Mat image_crop_padding(cv::Mat src, cv::Rect rect, cv::Scalar color = (0, 0, 0));


/***
 * 中心裁剪
 * @param src
 * @param crop_w
 * @param crop_h
 * @return
 */
cv::Mat image_center_crop(cv::Mat &src, int crop_w, int crop_h);


/***
 * 显示图像
 * @param name
 * @param image
 * @param waitKey
 */
void image_show(string name, cv::Mat image, int waitKey = 0);


/***
 * 保存图像
 * @param name
 * @param image
 */
void image_save(string name, cv::Mat image);


/***
 * 绘制点和文本
 *  image 图像
 *  center  圆心坐标
 *  radius 圆形的半径
 *  color 线条的颜色
 *  thickness 如果是正数，表示组成圆的线条的粗细程度。否则，表示圆是否被填充
 *  line_type 线条的类型。见 cvLine 的描述
 *  shift 圆心坐标点和半径值的小数点位数
 * @param image
 * @param point
 * @param text
 */
void draw_point_text(cv::Mat &image, cv::Point2d points, string text = "");


/***
 * 绘制多个点和文本
 * @param image
 * @param points
 * @param texts
 */
void draw_points_texts(cv::Mat &image, vector<cv::Point> points, vector<string> texts = {});


/***
 * 绘制矩形框
 * @param image
 * @param rect
 * @param text
 */
void draw_rect_text(cv::Mat &image, cv::Rect rect, string text = "");


/***
 * 绘制多个矩形框和文本
 * @param image
 * @param rects
 * @param texts
 */
void draw_rects_texts(cv::Mat &image, vector<cv::Rect> rects, vector<string> texts);


/***
 * 绘制连接线
 * @param image
 * @param points
 * @param skeleton 需要连接的ID序号
 */
void draw_lines(cv::Mat &image, vector<cv::Point> points, const vector<vector<float>> skeleton);

/***
 * 绘制带箭头的连接线
 * @param image
 * @param points
 * @param skeleton 需要连接的ID序号
 */
void draw_arrowed_line(cv::Mat &image, vector<cv::Point> points, const vector<vector<float>> skeleton);

#endif //DETECTOR_IMAGE_UTILS_H
