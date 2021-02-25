#include<opencv2/opencv.hpp>
#include<string>
#include "debug.h"
#include "image_utils.h"
#include "file_utils.h"

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

int main() {
    //test_opencv();
    //test_read_dir();
    test_read_write_file();
    return 0;
}
