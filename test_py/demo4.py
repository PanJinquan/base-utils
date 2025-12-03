# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2025-07-08 14:10:15
# @Brief  : 转换labelme标注数据为voc格式
# --------------------------------------------------------
"""
import os
import numpy as np
import cv2
from pybaseutils.converter import convert_labelme2voc
from pybaseutils import time_utils, image_utils, file_utils, json_utils


def gradio2openai_style(msgs):
    """
    gradio风格转换为openai风格
    :param msgs:
    :return:
    """
    out = []
    for msg in msgs:
        content, role = msg["content"], msg["role"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            tmps = []
            for file in content:
                if file_utils.is_image(file):
                    tmps.append({"type": "image", "image": file})
                elif file_utils.is_video(file):
                    tmps.append({"type": "video", "video": file})
            content = tmps
        if out and out[-1]["role"] == role:  # 合并相邻相同角色的消息
            out[-1]["content"].extend(content)
        else:
            msg["content"] = content
            out.append(msg)
    return out


def openai2gradio_style(msgs):
    """
    openai风格转换为gradio风格
    :param msgs:
    :return:
    """
    out = []
    for msg in msgs:
        content, role = msg["content"], msg["role"]
        if isinstance(content, list):  # 合并相邻相同角色的消息
            for c in content:
                if c["type"] == "text":
                    out.append({"role": role, "content": c["text"]})
                elif c["type"] == "image":
                    out.append({"role": role, "content": [c["image"]]})
                elif c["type"] == "video":
                    out.append({"role": role, "content": [c["video"]]})
        else:
            msg["content"] = content
            out.append(msg)
    return out


if __name__ == "__main__":
    image_file = "/media/PKing/dev2/SDK/base-utils/data/mask.png"
    msgs = [{"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，有什么我可以帮助你的吗？"},
            {"role": "user", "content": [image_file]},
            {"role": "user", "content": "请根据图片描述"},
            {"role": "assistant", "content": "这张图是描述....."},
            ]
    out1 = gradio2openai_style(msgs)
    out2 = openai2gradio_style(out1)
    print(json_utils.formatting(out2))
