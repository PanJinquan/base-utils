# base-utils

- 开源不易,麻烦给个【Star】
- 源码 ： https://github.com/PanJinquan/base-utils
- 安装包： https://pypi.org/project/pybaseutils/

## base_utils(C++)

base_util是个人开发常用的C++库，集成了C/C++ OpenCV等常用的算法

- 增加了debug测试宏定义，如时间测试，LOG信息等
- 针对目标坐标点的卡尔曼滤波，加权平均滤波
- 常用的文件处理函数
- 常用的OpenCV图像处理函数

## pybaseutils(Python)

pybaseutils是个人开发常用的python库，集成了python等常用的算法

- 安装方法1：pip install pybaseutils (有延时，可能不是最新版本)
- 安装方法2：pip install --upgrade pybaseutils -i https://pypi.org/simple (从pypi源下载最新版本)


## 一些问题修复说明

- [问题修复说明](./docs/README.md)

## 目录结构

```
├── base_utils         # base_utils的C++源代码
├── pybaseutils        # pybaseutils的python源代码
├── data               # 相关测试数据
├── test               # base_utils的测试代码
│   ├── build.sh
│   ├── CMakeLists.txt
│   ├── kalman_test.cpp
│   └── main.cpp
└── README.md

```

# base_utils(C++)

## 1.相关配置

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

- 配置OpenCL（可选）
    1. OpenCL: https://software.intel.com/content/www/us/en/develop/tools/opencl-sdk/choose-download.html
    2. Android系统一般都支持OpenCL，Linux系统可参考如下配置：

```bash
# 参考安装OpenCL： https://blog.csdn.net/qq_28483731/article/details/68235383，作为测试，安装`intel cpu版本的OpenCL`即可
# 安装clinfo，clinfo是一个显示OpenCL平台和设备的软件
sudo apt-get install clinfo
# 安装依赖
sudo apt install dkms xz-utils openssl libnuma1 libpciaccess0 bc curl libssl-dev lsb-core libicu-dev
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF
echo "deb http://download.mono-project.com/repo/debian wheezy main" | sudo tee /etc/apt/sources.list.d/mono-xamarin.list
sudo apt-get update
sudo apt-get install mono-complete
# 在intel官网上下载了intel SDK的tgz文件，并且解压
sudo sh install.sh
```

## 2.Demo测试

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
