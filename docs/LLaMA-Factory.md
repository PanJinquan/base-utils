# LLaMA-Factory

- https://github.com/hiyouga/LLaMA-Factory

## 安装

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

- modelscope下载模型默认地址：/home/datauser/.cache/modelscope/hub/models/Qwen/Qwen2-VL-2B-Instruct
-

## 运行

```bash
export CUDA_VISIBLE_DEVICES=0 # 指定运行GPU
export GRADIO_SERVER_PORT=30000 # 指定gradio的端口
export GRADIO_TEMP_DIR="~/.cache/gradio" # 指定gradio临时缓存路径，解决上传图片权限的问题
llamafactory-cli webui
```

