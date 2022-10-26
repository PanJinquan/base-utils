#include "stdio.h"
#include "ctypes_utils.h"


uchar *ct_imread(char *filename, int *rows, int *cols, int *dims) {
    cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);
    //dims =  image.channels();//错误写法
    *dims = image.channels();
    *rows = image.rows;
    *cols = image.cols;
    //uchar *output = image.data; // 错误，需要重新开辟空间返回，image是临时遍历，调用完后会被回收
    uchar *output = (uchar *) malloc(sizeof(uchar) * image.rows * image.cols * image.channels());
    mat2buffer(image, output);
    return output;
}


void ct_blur(uchar *src_p, uchar *dst_p, int rows, int cols, int dims) {
    cv::Mat src;
    cv::Mat dst;
    buffer2mat(src_p, src, rows, cols, dims);
    cv::blur(src, dst, cv::Size(8, 8));
    //memcpy(dst_p, dst.data, dst.rows * dst.cols * dst.channels());
    mat2buffer(dst, dst_p);
}

void ct_resize(CTImage *src_p, CTImage *dst_p, int size_w, int size_h) {
    cv::Mat src, dst;
    buffer2mat(src_p->data, src, src_p->rows, src_p->cols, src_p->dims);
    cv::resize(src, dst, cv::Size(size_w, size_h));
    dst_p->rows = dst.rows;
    dst_p->cols = dst.cols;
    dst_p->dims = dst.channels();
    //dst_p->data = dst.data;//错误
    mat2buffer(dst, dst_p->data);
}