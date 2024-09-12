# CUDA，cuDNN安装方法
- CUDA_VISIBLE_DEVICES=7,6,5,4 python train.py
## 常用的命令

```bash
 du -ah --max-depth=1/                            # 查看文件大小
 cat /usr/local/cuda/version.txt                  # 查看cuda版本
 cat /usr/include/cudnn_version.h                 # deb方式安装，查看cudnn版本
 cat /usr/local/cuda/include/cudnn_version.h      # 库安装方式，查看cudnn版本
 whereis cudnn_version.h                          # 查找方法
```

## 卸载cuDNN

- 卸载cudnn: https://blog.csdn.net/qq_45779334/article/details/123168792

```bash
dpkg -l | grep cudnn # 查看
# 卸载
dpkg -r libcudnn8-samples
dpkg -r libcudnn8-dev
dpkg -r libcudnn8
```

## 安装cuDNN

- 下载Local Installer for Linux x86_64 (Tar)，地址：https://developer.nvidia.com/rdp/cudnn-archive
- 解压
- 复制

```bash

# cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive根目录
cp -r include/* /usr/local/cuda/include/ 
cp -r lib/libcudnn* /usr/local/cuda/lib64/ 
chmod a+r /usr/local/cuda/include/cudnn.h 
chmod a+r /usr/local/cuda/lib64/libcudnn*
# 修改完成后，让配置生效
sudo ldconfig
```

## 安装torch

```bash
pip install --default-timeout=1000000 torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install  --no-cache-dir torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

## 安装TensorRT

#### 方法1
- 下载：TensorRT-8.4.1.5.Linux.x86_64-gnu.cuda-11.6.cudnn8.4，然后进入文件进行拷贝

```bash
pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple pycuda==2022.1
pip install --no-cache-dir  python/tensorrt-8.4.1.5-cp38-none-linux_x86_64.whl 
pip install --no-cache-dir  uff/uff-0.6.9-py2.py3-none-any.whl
pip install --no-cache-dir  graphsurgeon/graphsurgeon-0.4.6-py2.py3-none-any.whl

# (1) 配置环境变量：这样系统就可以搜索到TensorRT库（测试好像有问题）
sudo gedit ~/.bashrc
export LD_LIBRARY_PATH=$PATH:/home/PKing/Downloads/TensorRT-8.2.5.1/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$PATH:/home/PKing/Downloads/TensorRT-8.2.5.1/lib::$LIBRARY_PATH
source ~/.bashrc # 激活

# (2) 也可以将TensorRT相关库直接拷贝到系统目录，这样就不用配置环境变量了 (测试正常)
sudo cp -r lib/* /usr/lib/
sudo cp -r include/* /usr/include/
sudo ldconfig # 修改完成后，让配置生效
```

- 《ubuntu18.04+cuda11.0+tensorrt8.4.2配置安装》https://blog.csdn.net/qq_17364791/article/details/126368563
- 《ubuntu显卡驱动怎么重新安装》http://t.zoukankan.com/haiyang21-p-12699593.html
- python版tensorrt安装：https://algorithmic.blog.csdn.net/article/details/124640978
- 安装pycuda: pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple pycuda==2022.1
- Linux下的安装命令：

```bash
pip install tensorrt
pip install nvidia-pyindex
pip install nvidia-tensorrt==8.4.1.5 (版本與TensorRT-8.4.1.5.Linux.x86_64-gnu.cuda-11.6.cudnn8.4匹配即可)
```

#### 方法2

```bash
cd <tensorrt installation path>/python
pip install cuda-python
pip install tensorrt-8.6.0-cp310-none-win_amd64.whl
pip install opencv-python
```

## 制定GPU

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python train.py
```


## 没有root权限安装CUDA
- 参考：https://blog.csdn.net/qq_41105401/article/details/126038851 