cmake_minimum_required(VERSION 3.5)
project(base_utils)
set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -pthread")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g")



if (CMAKE_SYSTEM_NAME MATCHES "Android")
    add_definitions(-DPLATFORM_ANDROID)
elseif (CMAKE_SYSTEM_NAME MATCHES "Linux")
    add_definitions(-DPLATFORM_LINUX)
elseif (CMAKE_SYSTEM_NAME MATCHES "Windows")
    add_definitions(-DPLATFORM_WINDOWS)
endif ()

# 常用的变量：
# . 表示当前目录
# ${PROJECT_SOURCE_DIR}：为包含 project()的最近一个CMakeLists.txt文件所在的文件夹
# ${CMAKE_CURRENT_SOURCE_DIR}：当前CMakeLists.txt的路径

# opencv set
#find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS} ./src/)

# 指定头文件目录,PROJECT_SOURCE_DIR为工程的根目录  
include_directories(include)

#指定可执行文件的输出目录，输出到bin下面  
#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

#指定库文件输出路径  
#set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# 将指定的源文件生成链接文件base_utils
# add_library(base_utils myhello.cpp)
# 更加便捷的方法是使用aux_source_directory(<dir> <variable>)
# 查找当前目录下的所有源文件，并将名称保存到DIR_SRCS变量中
# DIR_SRCS可以多次使用，相同的变量会自动拼接
aux_source_directory(src DIR_SRCS)


#将指定的源文件生成链接文件
MESSAGE(STATUS "DIR_SRCS = ${DIR_SRCS}")
#add_library(base_utils STATIC ${DIR_SRCS}) # 静态库（.a、.lib）
#add_library(base_utils SHARED ${DIR_SRCS})  # 动态库（.so、.dll)
add_library(base_cl  ${DIR_SRCS})  #
#set_target_properties(my_lib PROPERTIES OUTPUT_NAME "my_lib")
