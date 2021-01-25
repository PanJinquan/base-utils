#include<opencv2/opencv.hpp>
#include<string>
#include "debug.h"

using namespace std;

int main() {
    string path = "../../data/test_image/test1.jpg";
    DEBUG_TIME(t1);
    cv::Mat image = cv::imread(path);
    LOGI("image:%s", path.c_str());
    LOGD("image:%s", path.c_str());
    LOGW("image:%s", path.c_str());
    LOGE("image:%s", path.c_str());
    LOGF("image:%s", path.c_str());
    DEBUG_TIME(t2);
    LOGI("rum time:%3.3fms", RUN_TIME(t2 - t1));
    cv::waitKey(0);
    DEBUG_IMSHOW("image", image);
    return 0;
}
