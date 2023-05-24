//
// Created by Pan on 2021/1/15.
//
#include<opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "image_utils.h"
#include "math_utils.h"


bool get_video_capture(string video_file, cv::VideoCapture &cap, int width, int height, int fps) {
#ifndef PLATFORM_ANDROID
    //VideoCapture video_cap;
    cap.open(video_file);
    if (width > 0 && height > 0) {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, width); //设置图像的宽度
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, height); //设置图像的高度
    }
    if (fps > 0) {
        cap.set(cv::CAP_PROP_FPS, fps);
    }
    if (!cap.isOpened())//判断是否读取成功
    {
        return false;
    }
#endif
    return true;
}

bool get_video_capture(int camera_id, cv::VideoCapture &cap, int width, int height, int fps) {
#ifndef PLATFORM_ANDROID
    //VideoCapture video_cap;
    cap.open(camera_id);    //摄像头ID号，默认从0开始
    if (width > 0 && height > 0) {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, width); //设置捕获图像的宽度
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);  //设置捕获图像的高度
    }
    if (fps > 0) {
        cap.set(cv::CAP_PROP_FPS, fps);
    }
    if (!cap.isOpened()) //判断是否成功打开相机
    {
        return false;
    }
#endif
    return true;
}


int VideoCaptureDemo(string video_file) {
#ifndef PLATFORM_ANDROID

    cv::VideoCapture cap;
    bool ret = get_video_capture(video_file, cap, 640, 480);
    cv::Mat frame;
    while (ret) {
        cap >> frame;
        if (frame.empty()) break;
        cv::imshow("frame", frame);
        if (27 == cv::waitKey(30)) {
            break;
        }
    }
    cap.release();         //释放对相机的控制
#endif
    return 0;
}


cv::Mat image_resize(cv::Mat &image, int resize_width, int resize_height) {
    cv::Mat dst;
    auto width = image.cols;
    auto height = image.rows;
    if (resize_height <= 0 && resize_width <= 0) {
        resize_width = width;
        resize_height = height;
    } else if (resize_height <= 0) {
        resize_height = int(height * resize_width / width);
    } else if (resize_width <= 0) {
        resize_width = int(width * resize_height / height);
    }
    cv::resize(image, dst, cv::Size(resize_width, resize_height));
    return dst;
}


cv::Mat rotate_image(cv::Mat &image, cv::Point2f center, float angle, cv::Scalar color) {
    //输出图像的尺寸与原图一样
    cv::Size dsize(image.cols, image.rows);
    //指定旋转中心
    //cv::Point2f center(image.cols / 2., image.rows / 2.);
    //获取旋转矩阵（2x3矩阵）
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Mat dst;
    cv::warpAffine(image, dst, rot_mat, dsize, cv::INTER_LINEAR, cv::BORDER_CONSTANT, color);
    return dst;
}

vector<cv::Point2f>
rotate_image_points(cv::Mat &image, vector<cv::Point2f> &points, cv::Point2f center, float angle) {
    image = rotate_image(image, center, angle);
    return rotate_points(points, center, image.cols, image.rows, angle);;
}

cv::Point2f rotate_point(cv::Point2f point, cv::Point2f center, int image_width, int image_height,
                         float angle) {
    // 将图像坐标转换到平面坐标
    float x1 = point.x;
    float y1 = image_height - point.y;
    float x2 = center.x;
    float y2 = image_height - center.y;
    float x = (x1 - x2) * cos(PI / 180.0 * angle) - (y1 - y2) * sin(PI / 180.0 * angle) + x2;
    float y = (x1 - x2) * sin(PI / 180.0 * angle) + (y1 - y2) * cos(PI / 180.0 * angle) + y2;
    // 将平面坐标转换到图像坐标
    y = image_height - y;
    return {x, y};
}

vector<cv::Point2f> rotate_points(vector<cv::Point2f> &points, cv::Point2f center,
                                  int image_width, int image_height, float angle) {
    vector<cv::Point2f> dst_points;
    for (auto &point:points) {
        dst_points.push_back(rotate_point(point, center, image_width, image_height, angle));
    }
    return dst_points;
}


cv::Rect extend_rect(cv::Rect rect, float sx, float sy) {
    float cx = (rect.x + rect.x + rect.width) / 2.0f;
    float cy = (rect.y + rect.y + rect.height) / 2.0f;
    float ew = rect.width * sx;
    float eh = rect.height * sy;
    float ex = cx - 0.5 * ew;
    float ey = cy - 0.5 * eh;
    cv::Rect r(ex, ey, ew, eh);
    return r;
}


cv::Mat image_crop(cv::Mat &image, cv::Rect rect) {
    cv::Mat dst;
    //求交集,避免越界
    rect &= cv::Rect(0, 0, image.cols, image.rows);
    image(rect).copyTo(dst);
    return dst;
};


cv::Mat image_crop(cv::Mat &image, int x1, int x2, int y1, int y2) {
    int width = image.cols;
    int height = image.rows;
    int left = std::max(0, (int) x1);
    int right = std::min((int) x2, width);
    int top = std::max(0, (int) y1);
    int bottom = std::min(int(y2), height);
    cv::Rect rect(left, top, right - left, bottom - top);
    cv::Mat dst = image_crop(image, rect);
    return dst;
};


cv::Mat image_crop_padding(cv::Mat &image, cv::Rect rect, cv::Scalar color) {
    int borderType = cv::BORDER_CONSTANT;//固定像素填充
    //int borderType = cv::BORDER_REPLICATE;//复制最边缘像素
    //int borderType = cv::BORDER_REFLECT_101;//边缘对称法填充
    int crop_x1 = cv::max(0, rect.x);
    int crop_y1 = cv::max(0, rect.y);
    int crop_x2 = cv::min(image.cols, rect.x + rect.width); // 图像范围 0到cols-1, 0到rows-1
    int crop_y2 = cv::min(image.rows, rect.y + rect.height);

    int left_x = (-rect.x);
    int top_y = (-rect.y);
    int right_x = rect.x + rect.width - image.cols;
    int down_y = rect.y + rect.height - image.rows;
    //cv::Mat roiImage = srcImage(cv::Range(crop_y1, crop_y2 + 1), cv::Range(crop_x1, crop_x2 + 1));
    cv::Mat roiImage = image(cv::Rect(crop_x1, crop_y1, (crop_x2 - crop_x1), (crop_y2 - crop_y1)));
    if (top_y > 0 || down_y > 0 || left_x > 0 || right_x > 0)//只要存在边界越界的情况，就需要边界填充
    {
        left_x = (left_x > 0 ? left_x : 0);
        right_x = (right_x > 0 ? right_x : 0);
        top_y = (top_y > 0 ? top_y : 0);
        down_y = (down_y > 0 ? down_y : 0);
        cv::copyMakeBorder(roiImage, roiImage, top_y, down_y, left_x, right_x, borderType, color);
    }
    return roiImage;
}

cv::Mat image_center_crop(cv::Mat &image, int crop_width, int crop_height) {
    auto width = image.cols;
    auto height = image.rows;
    int x1 = std::max(0, ((width - crop_width + 1) / 2));
    int y1 = std::max(0, ((height - crop_height + 1) / 2));
    int x2 = x1 + crop_width;
    int y2 = y1 + crop_height;
    cv::Mat dst = image_crop(image, x1, x2, y1, y2);
    return dst;
}


void image_show(string name, cv::Mat &image, int waitKey) {
#ifndef PLATFORM_ANDROID
    cv::namedWindow(name, cv::WINDOW_NORMAL);
    cv::Mat img_show = image.clone();
    if (img_show.channels() == 1)
        cvtColor(img_show, img_show, cv::COLOR_GRAY2BGR);
    //char str[200];
    //sprintf(str, ",Size:%dx%d", image.rows, image.cols);
    //RESIZE(img_show, 400);
    cv::imshow(name, img_show);
    cv::waitKey(waitKey);
#endif
}

void image_save(string name, cv::Mat &image) {
#ifndef PLATFORM_ANDROID
    cv::imwrite(name, image);
#endif

}


void draw_point_text(cv::Mat &image, cv::Point2f points, string text, cv::Scalar color) {
    int radius = 4;
    int thickness = -1;//实心点
    cv::circle(image, points, radius, color, thickness);
    if (text != "") {
        cv::putText(image,
                    text,
                    cv::Point(points.x + 5, points.y+20),
                    cv::FONT_HERSHEY_COMPLEX,
                    0.8,
                    color);
    }
}

void draw_points_texts(cv::Mat &image, vector<cv::Point2f> points, vector<string> texts,
                       cv::Scalar color) {
    int num = points.size();
    if (texts.size() != num && texts.size() == 0) {
        for (int i = 0; i < num; ++i) {
            texts.push_back("");
        }
    }
    for (int i = 0; i < num; ++i) {
        draw_point_text(image, points[i], texts[i], color);
    }
}


void draw_rect_text(cv::Mat &image, cv::Rect rect, string text, cv::Scalar color, int thickness,double fontScale) {
    cv::rectangle(image, rect, color, thickness);
    if (text != "") {
        cv::putText(image,
                    text,
                    cv::Point(rect.x + 5, rect.y - 5),
                    cv::FONT_HERSHEY_COMPLEX,
                    fontScale,
                    color, thickness);
    }
}

void draw_rects_texts(cv::Mat &image,
                      vector<cv::Rect> rects,
                      vector<string> texts,
                      cv::Scalar color,
                      int thickness,
                      double fontScale) {
    int num = rects.size();
    if (texts.size() != num && texts.size() == 0) {
        for (int i = 0; i < num; ++i) {
            texts.push_back("");
        }
    }
    for (int i = 0; i < num; ++i) {
        draw_rect_text(image, rects[i], texts[i], color, thickness,fontScale);
    }
}

void draw_lines(cv::Mat &image,
                vector<cv::Point2f> points,
                vector<vector<int>> skeleton,
                cv::Scalar color) {
    int thickness = 1;
    for (auto &pair:skeleton) {
        if (points[pair[0]].x > 0. && points[pair[0]].y > 0. &&
            points[pair[1]].x > 0. && points[pair[1]].y > 0.) {
            cv::Point2d p0 = points[pair[0]];
            cv::Point2d p1 = points[pair[1]];
            cv::line(image, p0, p1, color, thickness);
        }
    }
}

void draw_arrowed_lines(cv::Mat &image,
                        vector<cv::Point2f> points,
                        vector<vector<int>> skeleton,
                        cv::Scalar color) {
    int thickness = 1;
    for (auto &pair:skeleton) {
        if (points[pair[0]].x > 0. && points[pair[0]].y > 0. &&
            points[pair[1]].x > 0. && points[pair[1]].y > 0.) {
            cv::Point2d p0 = points[pair[0]];
            cv::Point2d p1 = points[pair[1]];
            cv::arrowedLine(image, p1, p0, color, thickness);
        }
    }
}


void draw_yaw_pitch_roll_in_left_axis(cv::Mat &imgBRG, float pitch, float yaw, float roll,
                                      cv::Point center, int size, int thickness, bool vis) {

    float cx = center.x;
    float cy = center.y;
    char text[200];
    sprintf(text, "(pitch,yaw,roll)=(%3.1f,%3.1f,%3.1f)", pitch, yaw, roll);
    pitch = pitch * PI / 180;
    yaw = -yaw * PI / 180;
    roll = roll * PI / 180;
    // X-Axis pointing to right. drawn in red
    float x1 = size * (cos(yaw) * cos(roll)) + cx;
    float y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + cy;
    cv::Scalar color_yaw_x(0, 0, 255); //BGR;
    // Y-Axis | drawn in green
    float x2 = size * (-cos(yaw) * sin(roll)) + cx;
    float y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + cy;
    cv::Scalar color_pitch_y(0, 255, 0);
    // Z-Axis (out of the screen) drawn in blue
    float x3 = size * (sin(yaw)) + cx;
    float y3 = size * (-cos(yaw) * sin(pitch)) + cy;
    cv::Scalar color_roll_z(255, 0, 0);
    float tipLength = 0.2;
    cv::arrowedLine(imgBRG, cv::Point(int(cx), int(cy)), cv::Point(int(x1), int(y1)), color_yaw_x,
                    thickness,
                    tipLength);
    cv::arrowedLine(imgBRG, cv::Point(int(cx), int(cy)), cv::Point(int(x2), int(y2)), color_pitch_y,
                    thickness,
                    tipLength);
    cv::arrowedLine(imgBRG, cv::Point(int(cx), int(cy)), cv::Point(int(x3), int(y3)), color_roll_z,
                    thickness,
                    tipLength);
    if (vis) {
        cv::putText(imgBRG,
                    text,
                    cv::Point(cx, cy),
                    cv::FONT_HERSHEY_COMPLEX,
                    0.5,
                    (0, 0, 255));
    }
}


void image_fusion(cv::Mat &imgBGR, cv::Mat matte, cv::Mat &out, cv::Scalar bg) {
    // cv::Mat bgi = cv::Mat::zeros(imgBGR.size(), CV_8UC3)+bg;
    cv::Mat bgi(imgBGR.size(), CV_8UC3, bg);
    image_fusion(imgBGR, matte, out, bgi);
}


void image_fusion(cv::Mat &imgBGR, cv::Mat matte, cv::Mat &out, cv::Mat bg) {
    assert(matte.channels() == 1);
    out.create(imgBGR.size(), CV_8UC3);
    vector<float> ratio{(float) imgBGR.cols / bg.cols, (float) imgBGR.rows / bg.rows};
    float max_ratio = *max_element(ratio.begin(), ratio.end());
    if (max_ratio > 1.0) {
        cv::resize(bg, bg, cv::Size(int(bg.cols * max_ratio), int(bg.rows * max_ratio)));
    }
    bg = image_center_crop(bg, imgBGR.cols, imgBGR.rows);
    int n = imgBGR.channels();
    int h = imgBGR.rows;
    int w = imgBGR.cols * n;
    // 循环体外进行乘法和除法运算
    matte.convertTo(matte, CV_32FC1, 1.0 / 255, 0);
    for (int i = 0; i < h; ++i) {
        uchar *sptr = imgBGR.ptr<uchar>(i);
        uchar *dptr = out.ptr<uchar>(i);
        float *mptr = matte.ptr<float>(i);
        uchar *bptr = bg.ptr<uchar>(i);
        for (int j = 0; j < w; j += n) {
            //float alpha = mptr[j] / 255; //循环体尽量减少乘法和除法运算
            float alpha = mptr[j / 3];
            float _alpha = 1.f - alpha;
            dptr[j] = uchar(sptr[j] * alpha + bptr[j] * _alpha);
            dptr[j + 1] = uchar(sptr[j + 1] * alpha + bptr[j + 1] * _alpha);
            dptr[j + 2] = uchar(sptr[j + 2] * alpha + bptr[j + 2] * _alpha);
        }
    }
}


void image_fusion_cv(cv::Mat &imgBGR, cv::Mat matte, cv::Mat &out, cv::Mat bg) {
    if (matte.channels() == 1) {
        matte.convertTo(matte, CV_32FC1, 1.0 / 255, 0);
        cv::cvtColor(matte, matte, cv::COLOR_GRAY2BGR);
    } else {
        matte.convertTo(matte, CV_32FC3, 1.0 / 255, 0);
    }
    //out = imgBGR.clone();
    vector<float> ratio{(float) imgBGR.cols / bg.cols, (float) imgBGR.rows / bg.rows};
    float max_ratio = *max_element(ratio.begin(), ratio.end());
    if (max_ratio > 1.0) {
        cv::resize(bg, bg, cv::Size(int(bg.cols * max_ratio), int(bg.rows * max_ratio)));
    }
    bg = image_center_crop(bg, imgBGR.cols, imgBGR.rows);
    bg.convertTo(bg, CV_32FC3, 1, 0);
    imgBGR.convertTo(out, CV_32FC3, 1, 0);
    // Fix a Bug: 1 - alpha实质上仅有B通道参与计算，多通道时(B,G,R)，需改Scalar(1.0, 1.0, 1.0)-alpha
    // out = out.mul(alpha) + bgi.mul(1 - alpha);
    out = out.mul(matte) + bg.mul(cv::Scalar(1.0, 1.0, 1.0) - matte);
    out.convertTo(out, CV_8UC3, 1, 0);
}

cv::Mat image_boxes_resize_padding(cv::Mat &image, cv::Size input_size, cv::Scalar color) {
    vector<cv::Box> boxes;
    return image_boxes_resize_padding(image, input_size, boxes, color);
}

cv::Mat image_boxes_resize_padding(cv::Mat &image, cv::Size input_size, vector<cv::Box> &boxes,
                                   cv::Scalar color) {
    int height = image.rows;
    int width = image.cols;
    //float scale = min([input_size[0] / width, input_size[1] / height]);
    vector<float> scale_ = {(float) input_size.width / width, (float) input_size.height / height};
    float scale = scale_[0] > scale_[1] ? scale_[1] : scale_[0];
    vector<int> new_size{int(width * scale), int(height * scale)};
    int pad_w = input_size.width - new_size[0];
    int pad_h = input_size.height - new_size[1];
    int top = pad_h / 2;
    int bottom = pad_h - (pad_h / 2);
    int left = pad_w / 2;
    int right = pad_w - (pad_w / 2);
    cv::Mat out;
    cv::resize(image, out, cv::Size(new_size[0], new_size[1]));
    //out = cv2::copyMakeBorder(out, top, bottom, left, right, cv2.BORDER_CONSTANT, value = color)
    cv::copyMakeBorder(out, out, top, bottom, left, right, cv::BORDER_CONSTANT, color);
    //if not boxes is None and len(boxes) > 0:
    //boxes[:] = boxes[:] * scale
    //boxes[:] = boxes[:] + [left, top, left, top]
    for (int i = 0; i < boxes.size(); i++) {
        boxes[i].x1 = boxes[i].x1 * scale + left;
        boxes[i].y1 = boxes[i].y1 * scale + top;
        boxes[i].x2 = boxes[i].x2 * scale + left;
        boxes[i].y2 = boxes[i].y2 * scale + top;
    }
    return out;
}

void image_boxes_resize_padding_inverse(cv::Size image_size, cv::Size input_size,
                                        vector<cv::Box> &boxes, vector<cv::Point2f> &points) {
    int height = image_size.height;
    int width = image_size.width;
    //scale = min([input_size[0] / width, input_size[1] / height])
    vector<float> scale_ = {(float) input_size.width / width, (float) input_size.height / height};
    float scale = scale_[0] > scale_[1] ? scale_[1] : scale_[0];
    //new_size = [int(width * scale), int(height * scale)]
    vector<int> new_size{int(width * scale), int(height * scale)};
    int pad_w = input_size.width - new_size[0];
    int pad_h = input_size.height - new_size[1];
    int top = pad_h / 2;
    int bottom = pad_h - (pad_h / 2);
    int left = pad_w / 2;
    int right = pad_w - (pad_w / 2);
    //if not boxes is None and len(boxes) > 0:
    //boxes[:] = boxes[:] - [left, top, left, top]
    //boxes[:] = boxes[:] / scale
    for (int i = 0; i < boxes.size(); i++) {
        boxes[i].x1 = (boxes[i].x1 - left) / scale;
        boxes[i].y1 = (boxes[i].y1 - top) / scale;
        boxes[i].x2 = (boxes[i].x2 - left) / scale;
        boxes[i].y2 = (boxes[i].y2 - top) / scale;
    }
    for (int i = 0; i < points.size(); i++) {
        points[i].x = (points[i].x - left) / scale;
        points[i].y = (points[i].y - top) / scale;
    }
}


void image_mosaic(cv::Mat &image, cv::Rect rect, int radius) {
    //仅对矩形框区域进行像素修改。遍历矩形框区域像素，并对其进行修改
    if (radius<=0) return;
    int n = image.channels();
    rect &= cv::Rect(0, 0, image.cols, image.rows);
    int xmax = rect.x + rect.width;
    int ymax = rect.y + rect.height;
    for (int i = rect.y; i < ymax; i += radius) {
        uchar *ptr1 = image.ptr<uchar>(i);
        for (int j = rect.x; j < xmax; j += radius) {
            //将矩形框再细分为若干个小方块，依次对每个方块修改像素（相同方块赋予相同灰度值）
            //cv::Vec3b v = image.at<cv::Vec3b>(i, j);
            cv::Vec3b v(ptr1[n * j], ptr1[n * j + 1], ptr1[n * j + 2]);
            for (int y = i; (y < (radius + i)) && (y < ymax); y++) {
                uchar *ptr2 = image.ptr<uchar>(y);
                for (int x = j; (x < (radius + j)) && (x < xmax); x++) {
                    //if (x > xmax) continue;
                    //对矩形区域像素值进行修改，rgb三通道
                    ptr2[n * x] = v[0];
                    ptr2[n * x + 1] = v[1];
                    ptr2[n * x + 2] = v[2];
                }
            }
        }
    }
}

void image_mosaic(cv::Mat &image, vector<cv::Rect> rects, int radius) {
    for (int i = 0; i < rects.size(); i++) {
        image_mosaic(image, rects[i], radius);
    }
}


void image_blur(cv::Mat &image, cv::Rect rect, int radius, bool gaussian) {
    if (radius<=0) return;
    rect &= cv::Rect(0, 0, image.cols, image.rows);
    cv::Mat roi = image(rect);
    if (gaussian) {
        radius = radius % 2 ? radius : (radius - 1); //取奇数
        cv::GaussianBlur(roi, roi, cv::Size(radius, radius), 11, 11);
    } else {
        cv::blur(roi, roi, cv::Size(radius, radius));
    }
}

void image_blur(cv::Mat &image, vector<cv::Rect> rects, int radius, bool gaussian) {
    for (int i = 0; i < rects.size(); i++) {
        image_blur(image, rects[i], radius, gaussian);
    }
}


cv::Box rect2box(cv::Rect &rect) {
    cv::Box box = {(float) rect.x, (float) rect.y, float(rect.x + rect.width),
                   float(rect.y + rect.height)};
    return box;
}

cv::Rect box2rect(cv::Box &box) {
    cv::Rect rect = {(int) box.x1, (int) box.y1, int(box.x2 - box.x1), int(box.y2 - box.y1)};
    return rect;
}

void boxes2rects(vector<cv::Box> &boxes, vector<cv::Rect> &rects) {
    for (int i = 0; i < boxes.size(); i++) {
        rects.push_back(box2rect(boxes[i]));
    }
}

void rects2boxes(vector<cv::Rect> &rects, vector<cv::Box> &boxes) {
    for (int i = 0; i < rects.size(); i++) {
        boxes.push_back(rect2box(rects[i]));
    }
}


void clip(cv::Mat &src, float vmin, float vmax) {
    int h = src.rows;
    int w = src.cols;
    if (src.isContinuous() && src.isContinuous()) {
        h = 1;
        w = w * src.rows * src.channels();
    }
    for (int i = 0; i < h; i++) {
        float *sptr = src.ptr<float>(i);
        for (int j = 0; j < w; j++) {
            //*dptr++ = *sptr++;
            sptr[j] = sptr[j] < vmax ? sptr[j] : vmax;
            sptr[j] = sptr[j] > vmin ? sptr[j] : vmin;
        }
    }
}

void clip_min(cv::Mat &src, float th, float v) {
    int h = src.rows;
    int w = src.cols;
    if (src.isContinuous() && src.isContinuous()) {
        h = 1;
        w = w * src.rows * src.channels();
    }
    for (int i = 0; i < h; i++) {
        float *sptr = src.ptr<float>(i);
        for (int j = 0; j < w; j++) {
            //*dptr++ = *sptr++;
            sptr[j] = sptr[j] < th ? v : sptr[j];
        }
    }
}