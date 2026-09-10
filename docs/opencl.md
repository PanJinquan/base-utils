# opencl
OpenCL通常由GPU驱动提供，如NVIDIA/AMD/Intel，请根据自己设备选择安装OpenCL驱动
## NVIDIA GPU
NVIDIA GPU需要安装NVIDIA驱动和CUDA，安装时且已经包含OpenCL驱动
```bash
# 确保已安装 NVIDIA 驱动（nvidia-smi 能正常输出）
sudo apt update
sudo apt install -y ocl-icd-opencl-dev opencl-headers clinfo

# 验证 OpenCL 驱动是否已安装
clinfo  # "Number of platforms" 应返回 >= 1
```

## AMD GPU
```bash
# 使用 ROCm 或 AMDGPU-PRO 驱动
sudo apt install -y rocm-opencl-runtime ocl-icd-opencl-dev opencl-headers clinfo

# 或者使用开源 mesa（性能较低但兼容性好）
sudo apt install -y mesa-opencl-icd ocl-icd-opencl-dev opencl-headers clinfo
```


## Intel核显/独显
```bash
# Intel Compute Runtime
sudo apt install -y intel-opencl-icd ocl-icd-opencl-dev opencl-headers clinfo

```



