#include<opencv2/opencv.hpp>
#include<string>
#include "debug.h"
#include "image_utils.h"

using namespace std;

int main() {
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
    return 0;
}
