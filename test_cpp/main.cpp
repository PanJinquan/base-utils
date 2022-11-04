#include<opencv2/opencv.hpp>
#include<string>
#include "debug.h"
#include "image_utils.h"
#include "file_utils.h"
#include "math_utils.h"

using namespace std;


void test_opencv() {
    string path = "../../data/test_image/test1.jpg";
    DEBUG_TIME(t1);
    cv::Mat image = cv::imread(path);
    image = image_resize(image, -1, 300);
    LOGI("image:%s,w-h=[%d,%d]", path.c_str(), image.cols, image.rows);
    LOGD("image:%s,w-h=[%d,%d]", path.c_str(), image.cols, image.rows);
    LOGW("image:%s,w-h=[%d,%d]", path.c_str(), image.cols, image.rows);
    LOGE("image:%s,w-h=[%d,%d]", path.c_str(), image.cols, image.rows);
    LOGF("image:%s,w-h=[%d,%d]", path.c_str(), image.cols, image.rows);
    DEBUG_TIME(t2);
    LOGI("rum time:%3.3fms", RUN_TIME(t2 - t1));
    cv::waitKey(0);
    DEBUG_IMSHOW("image", image);
}


void test_mosaic() {
    string path = "../../data/test.png";
    DEBUG_TIME(t1);
    cv::Mat image = cv::imread(path);
    cv::Rect rect(140, 10, 100, 200);
    vector<cv::Rect> rects;
    rects.push_back(rect);
    int radius = 15;//马赛克强度
    //image_blur(image, rect, radius);
    image_mosaic(image, rect, radius);
    draw_rect_text(image, rect);
    LOGI("image:%s,w-h=[%d,%d]", path.c_str(), image.cols, image.rows);
    LOGD("image:%s,w-h=[%d,%d]", path.c_str(), image.cols, image.rows);
    LOGW("image:%s,w-h=[%d,%d]", path.c_str(), image.cols, image.rows);
    LOGE("image:%s,w-h=[%d,%d]", path.c_str(), image.cols, image.rows);
    LOGF("image:%s,w-h=[%d,%d]", path.c_str(), image.cols, image.rows);
    DEBUG_TIME(t2);
    LOGI("rum time:%3.3fms", RUN_TIME(t2 - t1));
    cv::waitKey(0);
    DEBUG_IMSHOW("image", image);
}


void test_rotate_points() {
    string path = "../../data/test_image/test1.jpg";
    DEBUG_TIME(t1);
    cv::Mat image = cv::imread(path);
    float angle = 45;
    //cv::Point2f center(image.cols / 2., image.rows / 2.);
    cv::Point2f center(image.cols / 2., image.rows / 2.);
    cv::Point2f point1(238., 305);
    cv::Point2f point2(265., 280);
    cv::Mat dst = image.clone();
    vector<cv::Point2f> points;
    points.push_back(point1);
    points.push_back(point2);
    vector<cv::Point2f> dst_point = rotate_image_points(dst, points, center, angle);
    draw_points_texts(image, points);
    draw_points_texts(dst, dst_point);
    DEBUG_IMSHOW("image", image, 10);
    DEBUG_IMSHOW("dst", dst);
}


void test_image_padding() {
    string path = "../../data/test_image/grid2.png";
    DEBUG_TIME(t1);
    cv::Mat image = cv::imread(path);
    int height = image.rows;
    int width = image.cols;
    cv::Box box = {100, 100, 200, 200};
    vector<cv::Box> boxes = {box};
    vector<cv::Rect> rects;
    boxes2rects(boxes, rects);
    draw_rects_texts(image, rects, {}, cv::Scalar(255, 0, 0));
    cv::Mat dst = image_boxes_resize_padding(image, cv::Size(400, 400), boxes);
    vector<cv::Rect> rects2;
    boxes2rects(boxes, rects2);
    draw_rects_texts(dst, rects2);

    image_boxes_resize_padding_inverse(cv::Size(width, height),
                                       cv::Size(400, 400),
                                       boxes);
    vector<cv::Rect> rects3;
    boxes2rects(boxes, rects3);
    draw_rects_texts(image, rects3, {}, cv::Scalar(0, 255, 0));
    DEBUG_IMSHOW("image", image, 10);
    DEBUG_IMSHOW("dst", dst, 0);
}

void test_read_dir() {
    //string image_dir = "../../data/test_image/test1.jpg";
    string image_dir = "../../base_utils";
    vector<string> image_list = get_files_list(image_dir);
    for (int i = 0; i < image_list.size(); ++i) {
        string path = image_list.at(i);
        string subname = get_subname(path);
        string name = get_basename(path);
        printf("path:%s\tsubname:%s\tname:%s\n", path.c_str(), subname.c_str(), name.c_str());
    }
    //PRINT_VECTOR("image_list:\n", image_list);
}

void test_read_write_file() {
    string path = "../../data/write_contents.txt";
    string image_dir = "../../base_utils";
    vector<string> image_list = get_files_list(image_dir);
    write_contents(path, image_list, true);
    image_list = read_contents(path);
    PRINT_VECTOR("image_list:", image_list);
}

void test_math_utils() {
    vector<float> src = {0.01, 0.2, 10.};
    vector<float> dst;
    int max_index = 0;
    float max_value = 0;
    softmax(src, dst, max_index, max_value);
    PRINT_VECTOR("src:", src);
    PRINT_VECTOR("dst:", dst);
    LOGD("max_index:%d", max_index);
    LOGD("max_value:%f", max_value);

}

void test_math_utils_vector() {
    cv::Point2f point1(0, 0);
    cv::Point2f point2(1, 1);
    cv::Point2f point3(0, 0);
    cv::Point2f point4(1, 1);
    cv::Point2f v1 = create_vector(point1, point2);
    cv::Point2f v2 = create_vector(point3, point4);
    float angle = vector_angle(v1, v2, true);
    LOGD("angle:%f", angle);
}


int main() {
    //test_opencv();
    //test_read_dir();
    //test_read_write_file();
    //test_math_utils();
    //test_rotate_points();
    //test_math_utils_vector();
    //test_image_padding();
    test_mosaic();
    return 0;
}
