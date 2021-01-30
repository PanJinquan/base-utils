cmake_minimum_required(VERSION 3.5)
project(base_utils)

set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -pthread")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g")


set(BASE_FILTER_ENABLE ON CACHE BOOL "" FORCE)


if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    # -DCMAKE_BUILD_TYPE=Debug
    # -DCMAKE_BUILD_TYPE=Release
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Debug)" FORCE)
endif ()

# opencv set
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ./src/)
#MESSAGE(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")

if (CMAKE_SYSTEM_NAME MATCHES "Android")
    add_definitions(-DPLATFORM_ANDROID)
elseif (CMAKE_SYSTEM_NAME MATCHES "Linux")
    add_definitions(-DPLATFORM_LINUX)
    add_definitions(-DDEBUG_ON)            # for WIN/Linux Log
    add_definitions(-DDEBUG_LOG_ON)        # for WIN/Linux Log
    add_definitions(-DDEBUG_IMSHOW_ON)    # for OpenCV show
elseif (CMAKE_SYSTEM_NAME MATCHES "Windows")
    add_definitions(-DPLATFORM_WINDOWS)
    add_definitions(-DDEBUG_ON)            # for WIN/Linux Log
    add_definitions(-DDEBUG_LOG_ON)        # for WIN/Linux Log
    add_definitions(-DDEBUG_IMSHOW_ON)    # for OpenCV show
endif ()

# base_utils
set(BASE_ROOT ../) # 设置base-utils所在的根目录
add_subdirectory(${BASE_ROOT}/base_utils/ base_build) # 添加子目录到build中
include_directories(${BASE_ROOT}/base_utils/include)
include_directories(${BASE_ROOT}/base_utils/src)
MESSAGE(STATUS "BASE_ROOT = ${BASE_ROOT}")

# Test
add_executable(main main.cpp)
add_executable(kalman kalman_test.cpp)

target_link_libraries(main base_utils ${OpenCV_LIBS} -lpthread)
target_link_libraries(kalman base_utils ${OpenCV_LIBS} -lpthread)
