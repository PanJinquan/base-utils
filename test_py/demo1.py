# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail :
    @Date   : 2024-02-05 18:19:18
    @Brief  :
"""
import os
import cv2
import numpy as np
import random
import json
from PIL import Image
from typing import Dict
from tqdm import tqdm
from pybaseutils import image_utils, file_utils, json_utils, base64_utils, time_utils
from pybaseutils.cvutils import video_utils
from pybaseutils.transforms import transform_utils
from pybaseutils.converter import build_voc, build_labelme
from pybaseutils.dataloader import parser_labelme
import xmltodict

import torch
import torch.nn as nn


class CMDLoss(nn.Module):
    """计算两个分布的中心矩差差异的Loss函数"""

    def __init__(self, num_channels, use_gpu=True):
        super(CMDLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_channels = num_channels
        self.register_buffer('one', torch.tensor(1.0))

    def forward(self, input, target):
        # 计算均值
        input_mean = torch.mean(input, dim=[2, 3], keepdim=True)
        target_mean = torch.mean(target, dim=[2, 3], keepdim=True)

        # 计算方差
        input_var = torch.mean((input - input_mean) ** 2, dim=[2, 3], keepdim=True)
        target_var = torch.mean((target - target_mean) ** 2, dim=[2, 3], keepdim=True)

        # 计算协方差
        input_mean_expand = input_mean.expand_as(input)
        target_mean_expand = target_mean.expand_as(target)
        cov = torch.mean((input - input_mean_expand) * (target - target_mean_expand), dim=[2, 3], keepdim=True)

        # 计算CMD
        c = torch.mean(self.one / self.num_channels * cov, dim=[2, 3], keepdim=True)
        d = torch.mean(self.one / self.num_channels * (input_var - target_var), dim=[2, 3], keepdim=True)
        cmd_loss = c / (d + 1e-5)

        return cmd_loss