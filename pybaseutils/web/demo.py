# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : 2025-02-27 16:03:39
# @Brief  : 基于pywebio实现图像处理web小工具
#           运行方法：python app_webio_image.py
# --------------------------------------------------------
"""
import gradio as gr
from pathlib import Path

import os
import shutil
from pathlib import Path
import gradio as gr


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.Text(value='1')
            gr.Text(value='2')
        with gr.Column():
            gr.Text(value='3')
            gr.Text(value='4')
    with gr.Row():
        with gr.Column():
            gr.Text(value='11')
            gr.Text(value='21')
        with gr.Column():
            gr.Text(value='31')
            gr.Text(value='41')
demo.launch()
