# opencl

- OpenCL（通常由GPU驱动提供，如NVIDIA/AMD/Intel）
- 若使用NVIDIA GPU，安装CUDA时已包含OpenCL
- 对于Intel/AMD，可能需额外安装： sudo apt install -y opencl-headers ocl-icd-opencl-dev
- 验证：clinfo（可选）： sudo apt install clinfo && clinfo

## 安装Intel opencl

```bash
#Intel 仓库密钥

wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | sudo gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

#Intel oneAPI 软件源

echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list


#更新软件包列表并安装 OpenCL 运行时
sudo apt update
sudo apt install intel-oneapi-runtime-opencl # 这个包包含了在 Intel CPU 上运行 OpenCL 程序所需的核心库。

#（可选）安装 OpenCL 开发包
# 如果你计划编写或编译 OpenCL 程序，还需要安装头文件和开发库：
sudo apt install intel-oneapi-opencl
```
-   安装完成后，重启电脑或注销当前会话，然后运行 clinfo 命令来检查 OpenCL 环境是否已正确配置。 如果一切顺利，你会在输出信息中看到：
    Platform Name 会显示 Intel(R) OpenCL 或类似信息。
    Device Name 会显示 Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz。
    会列出 CPU 的核心数（16个）、最大计算单元等信息。
-   如果系统提示 clinfo: command not found，可以先通过 sudo apt install clinfo 安装这个工具，然后再进行验证。





