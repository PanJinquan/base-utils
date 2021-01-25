# base_utils
集成C/C++ OpenCV常用的算法
- 增加了debug测试宏定义，如时间测试，LOG信息等
- 针对目标坐标点的卡尔曼滤波，加权平均滤波
- 常用的文件处理函数
- 常用的OpenCV图像处理函数

## 1.目录结构
```
├── base_utils         # base_utils的源代码
├── data               # 相关测试数据
├── test               # base_utils的测试代码
│   ├── build.sh
│   ├── CMakeLists.txt
│   ├── kalman_test.cpp
│   └── main.cpp
└── README.md

```

## 2.相关配置
- OpenCV配置方法
```cmake
# opencv set
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ./src/)
MESSAGE(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
```

- base-utils库的配置方法
```cmake
# base-utils
set(BASE_ROOT ../) # 设置base-utils所在的根目录
add_subdirectory(${BASE_ROOT}/base_utils/ base_build) # 添加子目录到build中
include_directories(${BASE_ROOT}/base_utils/include)
include_directories(${BASE_ROOT}/base_utils/src)
MESSAGE(STATUS "BASE_ROOT = ${BASE_ROOT}")
```



## 3.Demo测试

- `build`
```bash
cd test
bash build.sh
```
- `test/main.cpp`测试样例
```c++
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

```