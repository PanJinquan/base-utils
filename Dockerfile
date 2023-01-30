FROM docker.dm-ai.cn/algorithm-research/calligraphy/py3.6-cuda11.2-cudnn8-trt8.4-torch1.7:v1
# FROM docker.dm-ai.cn/algorithm-research/calligraphy/py3.8-cuda11.0-cudnn8-trt8.4-torch1.7:v1

WORKDIR /app
ADD app /app/
RUN pip install --upgrade pybaseutils
RUN pip install easydict
CMD ["python", "main.py"]
