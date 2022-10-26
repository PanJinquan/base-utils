#include "stdio.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

/***
 * 定义C++的结构体CTImage
 */
struct CTImage {
    int rows; // width
    int cols; // height
    int dims; // channels
    uchar *data; // 数据
};


/***
 * 将buffer数组数据转换为Mat
 * @param src_p
 * @param dst
 * @param rows
 * @param cols
 * @param dims
 */
static void buffer2mat(uchar *src_p, cv::Mat &dst, int rows, int cols, int dims) {
    dst = cv::Mat(cv::Size(cols, rows), CV_8UC3, src_p);
}

/***
 * 将Mat转换为buffer数组
 * @param src
 * @param dst_p
 */
static void mat2buffer(cv::Mat &src, uchar *dst_p) {
    //调用前，需要使用malloc给dst_p开辟空间
    //uchar *dst_p = (uchar *) malloc(sizeof(uchar) * src.rows * src.cols * src.channels());
    memcpy(dst_p, src.data, src.rows * src.cols * src.channels());
}

/***
 * 打印结构体CTImage信息
 */
extern "C" void printCTImage(CTImage image) {
    printf("CTImage (rows,cols,dims)=(%d,%d,%d)\n", image.rows, image.cols, image.dims);
}

/***
 * 读取图片接口
 * @param filename
 * @param rows 输出图像rows
 * @param cols 输出图像cols
 * @param dims 输出图像channels
 * @return
 */
extern "C" uchar *ct_imread(char *filename, int *rows, int *cols, int *dims);

/***
 * 对图片进行模糊接口
 * @param src_p 输入
 * @param dst_p 输出
 * @param rows 图像rows
 * @param cols 图像cols
 * @param dims 图像channels
 */
extern "C" void ct_blur(uchar *src_p, uchar *dst_p, int rows, int cols, int dims);


/***
 * 对图像进行缩放接口
 * @param src_p 输入
 * @param dst_p 输出
 * @param size_w 缩放宽度
 * @param size_h 缩放高度
 */
extern "C" void ct_resize(CTImage *src_p, CTImage *dst_p, int size_w, int size_h);