# -*- coding: utf-8 -*-
import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2

class DepthAnything():
    def __init__(self, model_file, encoder='vits'):
        self.encoder = 'vits'  # 可选: 'vits', 'vitb', 'vitl'
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = DepthAnythingV2(**self.model_configs[encoder])
        self.model.load_state_dict(torch.load(model_file, map_location='cpu'))
        self.model = self.model.to(self.device).eval()

    def predict(self, image_file):
        image = cv2.imread(image_file)  # BGR 格式
        with torch.no_grad():
            depth = self.model.infer_image(image)  # shape: (H, W)
        depth = (depth / depth.max() * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        return depth, depth_colored


if __name__ == "__main__":
    model_file = "weights/depth_anything_v2_vits.pth"
    image_file = "images/image1.jpg"
    infer = DepthAnything(model_file)
    results = infer.predict(image_file)
