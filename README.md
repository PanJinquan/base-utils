# base_utils
- 集成C/C++ OpenCV常用的算法

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



## 3.测试
```bash
cd test
bash build.sh
```