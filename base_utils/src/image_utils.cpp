//
// Created by dm on 2021/1/15.
//

#include "image_utils.h"
#include "math_utils.h"

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

vector<cv::Point2f> rotate_image_points(cv::Mat &image, vector<cv::Point2f> &points, cv::Point2f center, float angle) {
    image = rotate_image(image, center, angle);
    return rotate_points(points, center, image.cols, image.rows, angle);;
}

cv::Point2f rotate_point(cv::Point2f point, cv::Point2f center, int image_width, int image_height, float angle) {
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
    cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
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
    cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
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
                    cv::Point(points.x + 5, points.y),
                    cv::FONT_HERSHEY_COMPLEX,
                    0.5,
                    color);
    }
}

void draw_points_texts(cv::Mat &image, vector<cv::Point2f> points, vector<string> texts, cv::Scalar color) {
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


void draw_rect_text(cv::Mat &image, cv::Rect rect, string text, cv::Scalar color) {
    cv::rectangle(image, rect, color, 2);
    if (text != "") {
        cv::putText(image,
                    text,
                    cv::Point(rect.x + 5, rect.y),
                    cv::FONT_HERSHEY_COMPLEX,
                    0.5,
                    color);
    }
}

void draw_rects_texts(cv::Mat &image,
                      vector<cv::Rect> rects,
                      vector<string> texts,
                      cv::Scalar color) {
    int num = rects.size();
    if (texts.size() != num && texts.size() == 0) {
        for (int i = 0; i < num; ++i) {
            texts.push_back("");
        }
    }
    for (int i = 0; i < num; ++i) {
        draw_rect_text(image, rects[i], texts[i], color);
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
                                      cv::Point center, bool vis, int size) {

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
    cv::arrowedLine(imgBRG, cv::Point(int(cx), int(cy)), cv::Point(int(x1), int(y1)), color_yaw_x, 2,
                    tipLength);
    cv::arrowedLine(imgBRG, cv::Point(int(cx), int(cy)), cv::Point(int(x2), int(y2)), color_pitch_y, 2,
                    tipLength);
    cv::arrowedLine(imgBRG, cv::Point(int(cx), int(cy)), cv::Point(int(x3), int(y3)), color_roll_z, 2,
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



