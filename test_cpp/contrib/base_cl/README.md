# OpenCL基础

## OpenCL的基本流程
- 选择平台（clGetPlatformIDs）并创建上下文（clCreateContextFromType）
- 选择设备并创建命令队列（clCreateCommandQueue）
- 创建（clCreateProgramWithSource）和构建程序对象（clBuildProgram）
- 创建内核和内存对象（clCreateBuffer），将主机数据复制到设备上下文
- 运行内核排队（clEnqueueNDRangeKernel）
- 从内核读回结果（clEnqueueReadBuffer）