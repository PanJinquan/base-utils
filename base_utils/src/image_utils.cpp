//
// Created by dm on 2021/1/15.
//

#include "image_utils.h"

cv::Mat image_resize(cv::Mat &src, int resize_width, int resize_height) {
    cv::Mat dst;
    auto width = src.cols;
    auto height = src.rows;
    if (resize_height <= 0 && resize_width <= 0) {
        resize_width = width;
        resize_height = height;
    } else if (resize_height <= 0) {
        resize_height = int(height * resize_width / width);
    } else if (resize_width <= 0) {
        resize_width = int(width * resize_height / height);
    }
    cv::resize(src, dst, cv::Size(resize_width, resize_height));
    return dst;
}

cv::Mat image_crop(cv::Mat &src, cv::Rect rect) {
    cv::Mat dst;
    //求交集,避免越界
    rect &= cv::Rect(0, 0, src.cols, src.rows);
    src(rect).copyTo(dst);
    return dst;
};


cv::Mat image_crop(cv::Mat &src, int x1, int x2, int y1, int y2) {
    int width = src.cols;
    int height = src.rows;
    int left = std::max(0, (int) x1);
    int right = std::min((int) x2, width);
    int top = std::max(0, (int) y1);
    int bottom = std::min(int(y2), height);
    cv::Rect rect(left, top, right - left, bottom - top);
    cv::Mat dst = image_crop(src, rect);
    return dst;
};


cv::Mat image_crop_padding(cv::Mat src, cv::Rect rect, cv::Scalar color) {
    int borderType = cv::BORDER_CONSTANT;//固定像素填充
    //int borderType = cv::BORDER_REPLICATE;//复制最边缘像素
    //int borderType = cv::BORDER_REFLECT_101;//边缘对称法填充
    int crop_x1 = cv::max(0, rect.x);
    int crop_y1 = cv::max(0, rect.y);
    int crop_x2 = cv::min(src.cols, rect.x + rect.width); // 图像范围 0到cols-1, 0到rows-1
    int crop_y2 = cv::min(src.rows, rect.y + rect.height);

    int left_x = (-rect.x);
    int top_y = (-rect.y);
    int right_x = rect.x + rect.width - src.cols;
    int down_y = rect.y + rect.height - src.rows;
    //cv::Mat roiImage = srcImage(cv::Range(crop_y1, crop_y2 + 1), cv::Range(crop_x1, crop_x2 + 1));
    cv::Mat roiImage = src(cv::Rect(crop_x1, crop_y1, (crop_x2 - crop_x1), (crop_y2 - crop_y1)));
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

cv::Mat image_center_crop(cv::Mat &src, int crop_w, int crop_h) {
    auto width = src.cols;
    auto height = src.rows;
    int x1 = std::max(0, ((width - crop_w + 1) / 2));
    int y1 = std::max(0, ((height - crop_h + 1) / 2));
    int x2 = x1 + crop_w;
    int y2 = y1 + crop_h;
    cv::Mat dst = image_crop(src, x1, x2, y1, y2);
    return dst;
}


void image_show(string name, cv::Mat image, int waitKey) {
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

void image_save(string name, cv::Mat image) {
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



