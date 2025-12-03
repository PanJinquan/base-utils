# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : 2025-11-24 11:32:24
# @Brief  :
# --------------------------------------------------------
"""
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# 加载预训练的 MobileNetV2 模型
model = models.mobilenet_v2(pretrained=True)
model.eval()  # 设置为评估模式

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载图片并推理
img = Image.open("example.jpg")
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)

with torch.no_grad():
    output = model(batch_t)

# 输出预测结果（ImageNet 1000 类）
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities.topk(5))