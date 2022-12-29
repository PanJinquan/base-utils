#FROM docker.dm-ai.cn/algorithm-research/calligraphy/calligraphy:v0.17
FROM docker.dm-ai.cn/algorithm-research/calligraphy/py36-trt8.4-torch17:v1.2

WORKDIR /app
ADD app /app/
RUN pip install --upgrade pybaseutils
RUN pip install easydict
#CMD ["python", "main.py"]
