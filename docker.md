# Docker使用方法

## 镜像


```bash
docker images # 查看所有镜像
docker ps  -a # 查看所有容器
```

- nvidia-docker镜像：https://hub.docker.com/r/nvidia/cuda

```bash
# docker pull nvidia/cuda:11.2.0-cudnn8-devel-ubuntu18.04
image=nvidia/cuda:11.2.0-cudnn8-devel-ubuntu18.04 # nvidia-docker基础镜像
image=docker.dm-ai.cn/algorithm-research/py38-cuda11.2-cudnn8.1-ubuntu18.04:base
image=docker.dm-ai.cn/algorithm-research/py38-cuda11.2-cudnn8.1-ubuntu18.04:torch1.8.1
image=docker.dm-ai.cn/algorithm-research/py38-cuda11.2-cudnn8.1-ubuntu18.04:torch1.8.1-trt8.2
image=docker.dm-ai.cn/algorithm-research/py38-cuda11.2-cudnn8.1-ubuntu18.04:torch1.8.1-trt8.4
docker run -it --gpus all -p 40000:80 -v `pwd`:/app $image /bin/bash
```

## 将容器转换为镜像

```bash
docker commit -m "info" -a "panjinquan" container_id image_id:tag
docker push image_id:tag
```

## pip安装方法

```bash
pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
pip install --default-timeout=1000000 --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
```

## docker清除垃圾

```bash
 du -ah --max-depth=1/                            # 查看文件大小
 
 apt-get autoclean
 apt-get clean
 apt-get autoremove
 rm -rf ~/.cache
 pip cache purge # 清除 pip 缓存中的所有文件。
 conda clean --all #删除conda无用的包和缓存
```

## 一些异常处理

- Error response from daemon: could not select device driver "" with capabilities: [[gpu]]

> 解决方法：https://zhuanlan.zhihu.com/p/373493938?utm_id=0

- libgl.so.1: cannot open shared object file

> 解决方法：https://blog.csdn.net/weixin_42990464/article/details/125203404

```bash
apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
# 修改完成后，让配置生效
sudo ldconfig
```

