cmake_minimum_required(VERSION 3.5)
project(base_utils)
add_compile_options(-fPIC) # fix Bug: can not be used when making a shared object
set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -pthread")
set(BASE_FILTER_ENABLE ON CACHE BOOL "" FORCE)
set(BASE_OPENMP_ENABLE ON CACHE BOOL "" FORCE)  # Multi-Thread
if (NOT CMAKE_BUILD_TYPE)
    # -DCMAKE_BUILD_TYPE=Debug
    # -DCMAKE_BUILD_TYPE=Release
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Debug)" FORCE)
endif ()


# set openMP
if (BASE_OPENMP_ENABLE)
    FIND_PACKAGE(OpenMP REQUIRED)
    if (OPENMP_FOUND)
        # target_link_libraries可以不添加-fopenmp
        message(STATUS "OPENMP FOUND")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS"${CMAKE_EXE_LINKER_FLAGS}${OpenMP_EXE_LINKER_FLAGS}")
    else ()
        error("OpenMP Not Found.")
    endif ()
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


# base_cl
set(BASE_CL contrib/base_cl) # 设置base_cl所在的根目录
add_subdirectory(${BASE_CL}/ base_cl_build) # 添加子目录到build中
include_directories(${BASE_CL}/include)
include_directories(${BASE_CL}/src)
MESSAGE(STATUS "BASE_CL = ${BASE_CL}")


# Test
add_executable(main main.cpp)
add_executable(filter_test filter_test.cpp)
add_executable(openmp_test openmp_test.cpp)
add_executable(opencl_test opencl_test.cpp)
add_executable(stereo_match_bm opencv/stereo_match_bm.cpp)
add_executable(stereo_match_sgbm opencv/stereo_match_sgbm.cpp)


#target_link_libraries(main base_utils ${OpenCV_LIBS} -lpthread)
target_link_libraries(main base_utils ${OpenCV_LIBS})
target_link_libraries(filter_test base_utils ${OpenCV_LIBS})
target_link_libraries(openmp_test base_utils ${OpenCV_LIBS} -lpthread)
target_link_libraries(opencl_test base_utils ${OpenCV_LIBS} OpenCL  -lpthread)
target_link_libraries(stereo_match_bm base_utils ${OpenCV_LIBS})
target_link_libraries(stereo_match_sgbm base_utils ${OpenCV_LIBS})
