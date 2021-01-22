#include<opencv2/opencv.hpp>
#include<string>
#include "debug.h"

using namespace std;

int main() {
    string path = "../../data/test_image/test1.jpg";
    cv::Mat image = cv::imread(path);
    cv::waitKey(0);
    DEBUG_IMSHOW("image", image);
    LOGI("image:%s", path.c_str());
    LOGD("image:%s", path.c_str());
    LOGW("image:%s", path.c_str());
    LOGE("image:%s", path.c_str());
    LOGF("image:%s", path.c_str());
    return 0;
}
