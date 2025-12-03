# -*- coding: utf-8 -*-
import torch
import numpy as np
import torchvision.transforms as transforms
from typing import Tuple, Optional
from pytorchvideo.models import create_slowfast
from pytorchvideo.transforms import UniformTemporalSubsample, Normalize, ShortSideScale

class SlowFastInference:
    def __init__(self, model_depth: int = 50, num_classes: int = 400, alpha: int = 4, slow_frames: int = 8,
                 fast_frames: Optional[int] = None, size: int = 224):
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.alpha = alpha
        self.size = size
        self.slow_frames = slow_frames
        self.fast_frames = fast_frames or (slow_frames * alpha)
        # 创建 SlowFast 模型
        self.model = create_slowfast(model_depth=model_depth, model_num_class=num_classes,
                                     head_pool_kernel_sizes=((1, 7, 7), (1, 7, 7)))
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            UniformTemporalSubsample(self.fast_frames),  # 先按 fast 帧数采样
            transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),  # (T, H, W, C) -> (T, C, H, W)
            ShortSideScale(size=self.size),
            Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
        ])

    def preprocess(self, video: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        if video.shape[0] < self.fast_frames:
            raise ValueError(f"Video too short: has {video.shape[0]} frames, need at least {self.fast_frames}")
        video_tensor = torch.from_numpy(video).float()  # (T, H, W, C)
        video_tensor = self.transform(video_tensor)  # (T_fast, C, H, W)
        # 分离 slow 和 fast 路径
        slow_indices = torch.linspace(0, self.fast_frames - 1, self.slow_frames).long()
        slow_tensor = video_tensor[slow_indices]  # (T_slow, C, H, W)
        slow_input = slow_tensor.unsqueeze(0).permute(0, 1, 2, 3, 4).to(self.device)
        fast_input = video_tensor.unsqueeze(0).permute(0, 1, 2, 3, 4).to(self.device)
        return slow_input, fast_input

    def predict(self, video: np.ndarray) -> torch.Tensor:
        slow_input, fast_input = self.preprocess(video)
        with torch.no_grad():
            logits = self.model([slow_input, fast_input])  # 输入为 [slow, fast]
        logits = logits.squeeze(0).argmax().item()
        return logits

if __name__ == "__main__":
    model_file = "weights/slowfast_50.pth"
    # 模拟一个(64, 224, 224, 3)的视频片段（64帧，224x224分辨率）
    dummy_video = np.random.randint(0, 256, size=(64, 224, 224, 3)).astype(np.float32)
    infer = SlowFastInference()
    results = infer.predict(dummy_video)
