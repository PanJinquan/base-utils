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
 * @param image
 * @param resize_width 默认-1
 * @param resize_height 默认-1
 * @return
 */
cv::Mat image_resize(cv::Mat &image, int resize_width = -1, int resize_height = -1);


/***
 * 逆时针旋转图像
 * @param image 图像
 * @param center 旋转中心点
 * @param angle  旋转角度
 * @param color  设置背景边界颜色：(0, 0, 0)
 * @return
 */
cv::Mat rotate_image(cv::Mat &image, cv::Point2f center, float angle, cv::Scalar color = (0, 0, 0));


/***
 * 逆时针旋转图像和关键点
 * @param image 输入图像/输出旋转后的图像
 * @param points 输入需要旋转的关键点
 * @param center 旋转中心点
 * @param angle  旋转角度
 * @return
 */
vector<cv::Point2f> rotate_image_points(cv::Mat &image, vector<cv::Point2f> &points, cv::Point2f center, float angle);

/***
 * 逆时针旋转图像中点
 * @param point 图像中需要旋转的点
 * @param center 旋转中心点
 * @param image_width   原始图像的width
 * @param image_height  原始图像的height
 * @param angle 旋转角度
 * @return
 */
cv::Point2f rotate_point(cv::Point2f point, cv::Point2f center, int image_width, int image_height, float angle);

/***
 * 逆时针旋转图像中点
 * @param points 图像中需要旋转的点
 * @param center 旋转中心点
 * @param image_width   原始图像的width
 * @param image_height  原始图像的height
 * @param angle 旋转角度
 * @return
 */
vector<cv::Point2f> rotate_points(vector<cv::Point2f> &points, cv::Point2f center,
                                  int image_width, int image_height, float angle);


/***
 * 扩展rect的大小
 * @param rect
 * @param sx X方向扩展倍数
 * @param sy Y方向扩展倍数
 * @return
 */
cv::Rect extend_rect(cv::Rect rect, float sx = 1.0f, float sy = 1.0f);

/***
 * 图像裁剪,超出的区域会被丢弃
 * @param image
 * @param rect
 * @return
 */
cv::Mat image_crop(cv::Mat &image, cv::Rect rect);


/***
 * 图像裁剪,超出的区域会被丢弃
 * @param image
 * @param x1
 * @param x2
 * @param y1
 * @param y2
 * @return
 */
cv::Mat image_crop(cv::Mat &image, int x1, int x2, int y1, int y2);


/***
 * 图像裁剪,超出的区域会被填充
 * @param image
 * @param rect
 * @param color 填充的颜色，默认黑色 color = (0, 0, 0)
 * @return
 */
cv::Mat image_crop_padding(cv::Mat &image, cv::Rect rect, cv::Scalar color = (0, 0, 0));


/***
 * 中心裁剪
 * @param image
 * @param crop_width
 * @param crop_height
 * @return
 */
cv::Mat image_center_crop(cv::Mat &image, int crop_width, int crop_height);


/***
 * 显示图像
 * @param name
 * @param image
 * @param waitKey
 */
void image_show(string name, cv::Mat &image, int waitKey = 0);


/***
 * 保存图像
 * @param name
 * @param image
 */
void image_save(string name, cv::Mat &image);


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
void draw_point_text(cv::Mat &image, cv::Point2f points, string text = "", cv::Scalar color = (0, 0, 255));


/***
 * 绘制多个点和文本
 * @param image
 * @param points
 * @param texts
 */
void draw_points_texts(cv::Mat &image, vector<cv::Point2f> points, vector<string> texts = {},
                       cv::Scalar color = (0, 0, 255));


/***
 * 绘制矩形框
 * @param image
 * @param rect
 * @param text
 */
void draw_rect_text(cv::Mat &image, cv::Rect rect, string text = "", cv::Scalar color = (255, 0, 0));


/***
 * 绘制多个矩形框和文本
 * @param image
 * @param rects
 * @param texts
 */
void draw_rects_texts(cv::Mat &image, vector<cv::Rect> rects, vector<string> texts, cv::Scalar color = (255, 0, 0));


/***
 * 绘制连接线
 * @param image
 * @param points
 * @param skeleton 需要连接的ID序号
 */
void
draw_lines(cv::Mat &image, vector<cv::Point2f> points, vector<vector<int>> skeleton, cv::Scalar color = (0, 255, 0));

/***
 * 绘制带箭头的连接线
 * @param image
 * @param points
 * @param skeleton 需要连接的ID序号
 */
void draw_arrowed_lines(cv::Mat &image, vector<cv::Point2f> points, vector<vector<int>> skeleton,
                        cv::Scalar color = (0, 255, 0));


/***
 * 绘制yaw,pitch,roll坐标轴(左手坐标系)
 * @param imgBRG 输入必须是BGR格式的图像
 * @param pitch红色X
 * @param yaw 绿色Y
 * @param roll 蓝色Z
 * @param center 坐标原始点
 * @param vis
 * @param size
 */
void draw_yaw_pitch_roll_in_left_axis(cv::Mat &imgBRG, float pitch, float yaw, float roll,
                                      cv::Point center, bool vis = true, int size = 50);

#endif //DETECTOR_IMAGE_UTILS_H
