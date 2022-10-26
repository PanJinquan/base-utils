cmake_minimum_required(VERSION 3.5)
project(Project)

add_compile_options(-fPIC) # fix Bug: can not be used when making a shared object
set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -pthread")

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

include_directories(src)
set(SRC_LIST src/ctypes_utils.cpp)
add_library(CTLib SHARED ${SRC_LIST})
target_link_libraries(CTLib ${OpenCV_LIBS})
MESSAGE(STATUS "DIR_SRCS = ${SRC_LIST}")

add_executable(CPPDemo main.cpp)
target_link_libraries(CPPDemo CTLib)
