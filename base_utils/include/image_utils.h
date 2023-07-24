//
// Created by Pan on 2021/1/15.
//

#ifndef DETECTOR_IMAGE_UTILS_H
#define DETECTOR_IMAGE_UTILS_H

#include <string>
#include "opencv2/opencv.hpp"

using namespace std;


namespace cv {
    struct Box {
        float x1, y1, x2, y2, score;   // 左上角x1,左上角y1,右下角x2,右下角y2,置信度分数score
    };
    static vector<Box> boxes = {};
    static vector<cv::Point2f> points = {};
}

// cv::Scalar c = COLOR_TABLE[obj.label%COLOR_TABLE.size()];
static vector<cv::Scalar> COLOR_TABLE = {
        {0,   0,   255},
        {0,   255, 0},
        {255, 0,   0},
        {128, 0,   0},
        {0,   128, 0},
        {128, 128, 0},
        {0,   0,   128},
        {128, 0,   128},
        {0,   128, 128},
        {128, 128, 128},
        {64,  0,   0},
        {192, 0,   0},
        {64,  128, 0},
        {192, 128, 0},
        {64,  0,   128},
        {192, 0,   128},
        {64,  128, 128},
        {192, 128, 128},
        {0,   64,  0},
        {128, 64,  0},
        {0,   192, 0},
        {128, 192, 0},
        {0,   64,  128}
};


/***
 * 读取视频文件
 * @param video_file 视频文件
 * @param cap 视频流对象
 * @param width 设置图像的宽度
 * @param height 设置图像的高度
 * @param fps 设置视频播放频率
 * @return
 */
bool get_video_capture(string video_file, cv::VideoCapture &cap, int width = -1, int height = -1, int fps = -1);


/***
 * 读取摄像头
 * @param camera_id 摄像头ID号，默认从0开始
 * @param cap 视频流对象
 * @param width 设置图像的宽度
 * @param height 设置图像的高度
 * @param fps 设置视频播放频率
 * @return
 */
bool get_video_capture(int camera_id, cv::VideoCapture &cap, int width = -1, int height = -1, int fps = -1);


/***
 * 测试demo视频文件
 * @return
 */
int VideoCaptureDemo(string video_file);


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
cv::Mat rotate_image(cv::Mat &image, cv::Point2f center, float angle, cv::Scalar color = cv::Scalar(0, 0, 0));


/***
 * 逆时针旋转图像和关键点
 * @param image 输入图像/输出旋转后的图像
 * @param points 输入需要旋转的关键点
 * @param center 旋转中心点
 * @param angle  旋转角度
 * @return
 */
vector<cv::Point2f>
rotate_image_points(cv::Mat &image, vector<cv::Point2f> &points, cv::Point2f center, float angle);

/***
 * 逆时针旋转图像中点
 * @param point 图像中需要旋转的点
 * @param center 旋转中心点
 * @param image_width   原始图像的width
 * @param image_height  原始图像的height
 * @param angle 旋转角度
 * @return
 */
cv::Point2f
rotate_point(cv::Point2f point, cv::Point2f center, int image_width, int image_height, float angle);

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
cv::Mat image_crop_padding(cv::Mat &image, cv::Rect rect, cv::Scalar color = cv::Scalar(0, 0, 0));


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
void draw_point_text(cv::Mat &image, cv::Point2f points, string text = "",
                     cv::Scalar color = cv::Scalar(255, 0, 0));


/***
 * 绘制多个点和文本
 * @param image
 * @param points
 * @param texts
 */
void draw_points_texts(cv::Mat &image, vector<cv::Point2f> points, vector<string> texts = {},
                       cv::Scalar color = cv::Scalar(255, 0, 0));


/***
 * 绘制矩形框
 * @param image
 * @param rect
 * @param text
 */
void draw_rect_text(cv::Mat &image, cv::Rect rect, string text = "",
                    cv::Scalar color = cv::Scalar(255, 0, 0), int thickness = 2, double fontScale = 0.8);


/***
 * 绘制多个矩形框和文本
 * @param image
 * @param rects
 * @param texts
 */
void draw_rects_texts(cv::Mat &image, vector<cv::Rect> rects, vector<string> texts = {},
                      cv::Scalar color = cv::Scalar(255, 0, 0), int thickness = 2, double fontScale = 0.8);


/***
 * 绘制连接线
 * @param image
 * @param points
 * @param skeleton 需要连接的ID序号
 */
void draw_lines(cv::Mat &image, vector<cv::Point2f> points, vector<vector<int>> skeleton,
                cv::Scalar color = cv::Scalar(255, 0, 0));

/***
 * 绘制带箭头的连接线
 * @param image
 * @param points
 * @param skeleton 需要连接的ID序号
 */
void draw_arrowed_lines(cv::Mat &image, vector<cv::Point2f> points, vector<vector<int>> skeleton,
                        cv::Scalar color = cv::Scalar(255, 0, 0));


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
                                      cv::Point center, int size = 50, int thickness = 2,
                                      bool vis = true);


/***
 * 实现图像融合：out = imgBGR * matte + bg * (1 - matte)
 * Fix a Bug: 1-alpha实质上仅有B通道参与计算，多通道时(B,G,R)，需改Scalar(1.0, 1.0, 1.0)-alpha
 * @param imgBGR 输入原始图像
 * @param matte  输入原始图像的Mask,或者alpha,matte
 * @param out    输出融合图像
 * @param bg     输入背景图像Mat(可任意大小)，也可以通过Scalar指定纯色的背景
 */
void image_fusion(cv::Mat &imgBGR, cv::Mat matte, cv::Mat &out,
                  cv::Scalar bg = cv::Scalar(219, 142, 67));

/***
 * 对图像进行融合
 * @param imgBGR
 * @param matte
 * @param out
 * @param bg
 */
void image_fusion(cv::Mat &imgBGR, cv::Mat matte, cv::Mat &out, cv::Mat bg);

/***
 * 对图像进行融合
 * @param imgBGR
 * @param matte
 * @param out
 * @param bg
 */
void image_fusion_cv(cv::Mat &imgBGR, cv::Mat matte, cv::Mat &out, cv::Mat bg);


/***
 * 对图像进行等比例缩放和填充
 * @param image
 * @param input_size
 * @param color
 * @return
 */
cv::Mat image_boxes_resize_padding(cv::Mat &image, cv::Size input_size, cv::Scalar color = cv::Scalar(0, 0, 0));

/***
 * 对图像进行等比例缩放和填充
 * @param image
 * @param input_size
 * @param boxes
 * @param color
 * @return
 */
cv::Mat
image_boxes_resize_padding(cv::Mat &image, cv::Size input_size, vector<cv::Box> &boxes,
                           cv::Scalar color = cv::Scalar(0, 0, 0));

/****
 * image_boxes_resize_padding的逆过程
 * @param image_size
 * @param input_size
 * @param boxes
 */
void image_boxes_resize_padding_inverse(cv::Size image_size, cv::Size input_size,
                                        vector<cv::Box> &boxes = cv::boxes,
                                        vector<cv::Point2f> &points = cv::points);


/***
 * 对图像进行马赛克处理
 * @param image 输入图像
 * @param rect 马赛克区域
 * @param radius 马赛克强度
 */
void image_mosaic(cv::Mat &image, cv::Rect rect, int radius = 5);

/***
 * 对图像进行马赛克处理
 * @param image 输入图像
 * @param rects 马赛克区域集合
 * @param radius 马赛克强度
 */
void image_mosaic(cv::Mat &image, vector<cv::Rect> rects, int radius = 5);


/***
 * 对图像进行模糊处理
 * @param image 输入图像
 * @param rect 模糊处理区域
 * @param radius 模糊强度
 * @param gaussian 是否使用高斯模糊，默认均值模糊
 */
void image_blur(cv::Mat &image, cv::Rect rect, int radius = 5, bool gaussian = false);

/***
 * 对图像进行模糊处理
 * @param image 输入图像
 * @param rects 模糊处理区域集合
 * @param radius 模糊强度
 * @param gaussian 是否使用高斯模糊，默认均值模糊
 */
void image_blur(cv::Mat &image, vector<cv::Rect> rects, int radius = 5, bool gaussian = false);


cv::Box rect2box(cv::Rect &rect);

cv::Rect box2rect(cv::Box &box);

void boxes2rects(vector<cv::Box> &boxes, vector<cv::Rect> &rects);

void rects2boxes(vector<cv::Rect> &rects, vector<cv::Box> &boxes);

/***
 * 限制最大最小值
 * @param src
 * @param vmin
 * @param vmax
 */
void clip(cv::Mat &src, float vmin, float vmax);

/***
 * 限制最大最小值
 * @param src
 * @param th
 * @param vmin
 */
void clip_min(cv::Mat &src, float th, float vmin);


#endif //DETECTOR_IMAGE_UTILS_H
