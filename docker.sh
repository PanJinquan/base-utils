#!/usr/bin/env bash

#image=docker.dm-ai.cn/algorithm-research/calligraphy/py3.8-cuda11.0-cudnn8-trt8.4-torch1.7:v2
#image=docker.dm-ai.cn/algorithm-research/py38-cuda11.2-cudnn8.1-ubuntu18.04:torch1.8.1-trt8.4
#sudo docker run -it --gpus all -p 40000:80 -v `pwd`:/app $image /bin/bash

############### 将容器转换为镜像 ###############
# docker commit -m "info" -a "author" container_id image_ids:tag
# docker push image_ids:tag

############### 一些异常处理    ###############
# Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
# 解决方法：https://zhuanlan.zhihu.com/p/373493938?utm_id=0
# pip安装：pip install --no-cache-dir opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple

############### docker清除垃圾 ###############
# apt-get autoclean
# apt-get clean
# apt-get autoremove
# rm -rf ~/.cache
# pip cache purge # 清除 pip 缓存中的所有文件。

############### 通过Dockerfile构建docker ###############
image="docker.dm-ai.cn/algorithm-research/py38-cuda11.2-cudnn8.1-ubuntu18.04:latest"
docker build -t=$image .
docker run -it --gpus all -p 40000:80 -v `pwd`:/app $image /bin/bash