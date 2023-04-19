//
//
// Created by 390737991@qq.com on 2022/10/6.
//
#include <opencv2/opencv.hpp>
#include <iostream>
#include "ctypes_utils.h"

using namespace std;


void test_dmmo() {
    string filename = "../test.png";
    cv::Mat img1 = cv::imread(filename);
    uchar *img1_p = (uchar *) malloc(sizeof(uchar) * img1.rows * img1.cols * img1.channels());
    mat2buffer(img1, img1_p);
    cv::Mat img2;
    buffer2mat(img1_p, img2, img1.rows, img1.cols, img1.channels());
    imshow("img1", img1);
    imshow("img2", img2);
    cv::waitKey(0);
}

void test_ct_dmmo() {
    string filename = "../test.png";
    int rows = 0;
    int cols = 0;
    int channels = 0;
    uchar *src_p = ct_imread(const_cast<char *>(filename.c_str()), &rows, &cols, &channels);
    cv::Mat src = cv::Mat(cv::Size(cols, rows), CV_8UC3, src_p);

    uchar *dst_p = (uchar *) malloc(sizeof(uchar) * src.rows * src.cols * src.channels());
    ct_blur(src.data, dst_p, rows, cols, channels);
    cv::Mat dst = cv::Mat(cv::Size(cols, rows), CV_8UC3, dst_p);
    imshow("src", src);
    imshow("dst", dst);
    cv::waitKey(0);
}

void test_ct_resize() {
    string filename = "../test.png";
    int rows = 0;
    int cols = 0;
    int channels = 0;
    int size_w = 200;
    int size_h = 400;
    uchar *src_p = ct_imread(const_cast<char *>(filename.c_str()), &rows, &cols, &channels);
    uchar *dst_p = (uchar *) malloc(sizeof(uchar) * rows * cols * channels);
    CTImage img1 = {rows, cols, channels, src_p};
    CTImage img2 = {0, 0, 0, dst_p};
    ct_resize(&img1, &img2, size_w, size_h);
    printCTImage(img1);
    printCTImage(img2);
    cv::Mat dst;
    buffer2mat(img2.data, dst, img2.rows, img2.cols, img2.dims);
    imshow("dst", dst);
    cv::waitKey(0);
}


int main() {
    //test_dmmo();
    //test_ct_dmmo();
    test_ct_resize();
    return 0;
}
