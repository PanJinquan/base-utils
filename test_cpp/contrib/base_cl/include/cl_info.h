//
// Created by dm on 2021/2/19.
//

#ifndef BASE_UTILS_CL_INFO_H
#define BASE_UTILS_CL_INFO_H

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else

#include <CL/cl.h>

#endif

int get_cl_info() {

    /* Host data structures */
    cl_platform_id *platforms;
    //每一个cl_platform_id 结构表示一个在主机上的OpenCL执行平台，就是指电脑中支持OpenCL的硬件，如nvidia显卡，intel CPU和显卡，AMD显卡和CPU等
    cl_uint num_platforms;
    cl_int i, err, platform_index = -1;

    /* Extension data */
    char *ext_data;
    size_t ext_size;
    const char icd_ext[] = "cl_khr_icd";

    //要使platform工作，需要两个步骤。1 需要为cl_platform_id结构分配内存空间。2 需要调用clGetPlatformIDs初始化这些数据结构。一般还需要步骤0：询问主机上有多少platforms

    /* Find number of platforms */
    //返回值如果为-1就说明调用函数失败，如果为0标明成功
    //第二个参数为NULL代表要咨询主机上有多少个platform，并使用num_platforms取得实际flatform数量。
    //第一个参数为1，代表我们需要取最多1个platform。可以改为任意大如：INT_MAX整数最大值。但是据说0，否则会报错，实际测试好像不会报错。下面是步骤0：询问主机有多少platforms
    err = clGetPlatformIDs(5, NULL, &num_platforms);
    if (err < 0) {
        perror("Couldn't find any platforms.");
        exit(1);
    }

    printf("I have platforms: %d\n", num_platforms); //本人计算机上显示为2，有intel和nvidia两个平台

    /* Access all installed platforms */
    //步骤1 创建cl_platform_id，并分配空间
    platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * num_platforms);
    //步骤2 第二个参数用指针platforms存储platform
    clGetPlatformIDs(num_platforms, platforms, NULL);

    /* Find extensions of all platforms */
    //获取额外的平台信息。上面已经取得了平台id了，那么就可以进一步获取更加详细的信息了。
    //一个for循环获取所有的主机上的platforms信息
    for (i = 0; i < num_platforms; i++) {
        /* Find size of extension data */
        //也是和前面一样，先设置第三和第四个参数为0和NULL，然后就可以用第五个参数ext_size获取额外信息的长度了。
        err = clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, 0, NULL, &ext_size);
        if (err < 0) {
            perror("Couldn't read extension data.");
            exit(1);
        }

        printf("The size of extension data is: %d\n", (int) ext_size);//我的计算机显示224.

        /* Access extension data */
        //这里的ext_data相当于一个缓存，存储相关信息。
        ext_data = (char *) malloc(ext_size);
        //这个函数就是获取相关信息的函数，第二个参数指明了需要什么样的信息，如这里的CL_PLATFORM_EXTENSIONS表示是opencl支持的扩展功能信息。我计算机输出一大串，机器比较新（专门为了学图形学而购置的电脑），支持的东西比较多。
        clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, ext_size, ext_data, NULL);
        printf("Platform %d supports extensions: %s\n", i, ext_data);

        //这里是输出生产商的名字，比如我显卡信息是：NVIDIA CUDA
        char *name = (char *) malloc(ext_size);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, ext_size, name, NULL);
        printf("Platform %d name: %s\n", i, name);

        //这里是供应商信息，我显卡信息：NVIDIA Corporation
        char *vendor = (char *) malloc(ext_size);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, ext_size, vendor, NULL);
        printf("Platform %d vendor: %s\n", i, vendor);

        //最高支持的OpenCL版本，本机显示：OpenCL1.1 CUDA 4.2.1
        char *version = (char *) malloc(ext_size);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, ext_size, version, NULL);
        printf("Platform %d version: %s\n", i, version);

        //这个只有两个值：full profile 和 embeded profile
        char *profile = (char *) malloc(ext_size);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, ext_size, profile, NULL);
        printf("Platform %d full profile or embeded profile?: %s\n", i, profile);

        /* Look for ICD extension */
        //如果支持ICD这一扩展功能的platform，输出显示，本机的Intel和Nvidia都支持这一扩展功能
        if (strstr(ext_data, icd_ext) != NULL)
            platform_index = i;
        //std::cout<<"Platform_index = "<<platform_index<<std::endl;
        printf("Platform_index is: %d\n", platform_index);
        /* Display whether ICD extension is supported */
        if (platform_index > -1)
            printf("Platform %d supports the %s extension.\n", platform_index, icd_ext);


        //释放空间
        free(ext_data);
        free(name);
        free(vendor);
        free(version);
        free(profile);
    }

    if (platform_index <= -1)
        printf("No platforms support the %s extension.\n", icd_ext);

    /* Deallocate resources */
    free(platforms);
    printf("done\n", icd_ext);
    return 0;
}

#endif //BASE_UTILS_CL_INFO_H
