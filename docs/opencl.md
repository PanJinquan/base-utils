# opencl

## 安装

- OpenCL（通常由GPU驱动提供，如NVIDIA/AMD/Intel）
- 若使用NVIDIA GPU，安装CUDA时已包含OpenCL
- 对于Intel/AMD，可能需额外安装： sudo apt install -y opencl-headers ocl-icd-opencl-dev
- 验证：clinfo（可选）： sudo apt install clinfo && clinfo
