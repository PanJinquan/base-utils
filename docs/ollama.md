# ollama教程

- ollama： https://ollama.com/download/linux:
- 安装命令： curl -fsSL https://ollama.com/install.sh | sh
- 局域网内配置 Ollama 服务以实现远程访问: https://www.11zhang.com/1456.html
- https://blog.csdn.net/qq_41297934/article/details/145612269
- /etc/systemd/system

## ollama镜像

- docker下运行ollama及deepseek: https://zhuanlan.zhihu.com/p/21303498630

```bash
# 安装依赖项
apt update && apt install -y tmux apciutils lshw
# 拉取镜像
docker pull ollama/ollama:latest
docker pull docker-0.unsee.tech/ollama/ollama:latest
# 启动dokcer http://192.168.68.102:50000/
docker run -d --gpus all  -p 50000:11434 -v `pwd`:/root/.ollama --name ollama docker-0.unsee.tech/ollama/ollama:latest
# 进入容器
docker exec -it ollama bash
# docker exec -it ollama ollama run deepseek-r1:1.5b
# 可以将本地模型文件：/usr/share/ollama/.ollama/models 或~/.ollama/models挂载到docker中/root/.ollama
# 复制制定模型：cp -r /usr/share/ollama/.ollama/models ~/project/ollama
# 复制单个模型可能不完整，docker无法检测到

docker pull docker-0.unsee.tech/nvidia/cuda:12.0.0-devel-ubuntu20.04
image=docker.dm-ai.cn/algorithm-research/panjinquan/ubuntu20.04-cuda12.0.0-ollama:base 
docker run -it --gpus all -p 40000:80 -v `pwd`:/app $image /bin/bash

```

## ollama常用命令

```bash
# 下载模型
# 列出本地可用的模型列表：
ollama list
# 启动模型(会自动下载模型)
ollama run model_name
ollama run deepseek-r1:1.5b
# 查看模型信息
ollama show model_name
# 删除指定模型：
ollama rm model_name
```

# 对话命令

```bash
  /set            Set session variables
  /show           Show model information
  /load <model>   Load a session or model
  /save <model>   Save your current session
  /clear          Clear session context
  /bye            Exit
  /?, /help       Help for a command
  /? shortcuts    Help for keyboard shortcuts
```