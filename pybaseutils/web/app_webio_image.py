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
import os
import cv2
import numpy as np
from pywebio.input import file_upload, input_group, actions
from pywebio.output import put_image, put_buttons, put_row, put_column, use_scope, clear, put_text, put_scope
from pywebio.pin import pin, put_input
from pywebio.session import set_env
from pywebio.pin import *


def blur_image(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    # 应用高斯模糊
    blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
    # 保存模糊后的图像
    blurred_image_path = f"blurred_{os.path.basename(image_path)}"
    cv2.imwrite(blurred_image_path, blurred_image)
    return blurred_image_path


def show_image(image_path):
    with use_scope('image_display', clear=True):
        put_image(open(image_path, 'rb').read())


def handle_single_image(file_info):
    file_name = file_info['filename']
    file_content = file_info['content']
    with open(file_name, 'wb') as f:
        f.write(file_content)
    blurred_image_path = blur_image(file_name)
    show_image(blurred_image_path)


def handle_multiple_images(directory):
    global current_image_index
    current_image_index = 0
    images = [os.path.join(directory, f) for f in os.listdir(directory) if
              f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        put_text("No images found in the selected directory.")
        return
    current_image_path = images[current_image_index]
    blurred_image_path = blur_image(current_image_path)
    show_image(blurred_image_path)
    put_buttons(['Previous', 'Next'], onclick=[lambda: switch_image(images, -1), lambda: switch_image(images, 1)])


def switch_image(images, direction):
    global current_image_index
    current_image_index += direction
    current_image_index = max(0, min(current_image_index, len(images) - 1))
    current_image_path = images[current_image_index]
    blurred_image_path = blur_image(current_image_path)
    show_image(blurred_image_path)


def main():
    set_env(output_max_width='80%')
    put_column([
        put_row([
            put_buttons(['选择图片'], onclick=lambda: pin.update(
                name=file_upload(label='选择图片', accept='image/*', help_text='请选择一张图片'))),
            put_buttons(['选择目录'], onclick=lambda: pin.update(name=input_group("选择目录", [
                input('directory', type='text', name='directory', placeholder='请输入目录路径')]))),
        ]),
        put_scope('image_display')
    ])

    while True:
        event = pin_wait_change('name')
        if event['name'] == 'value':
            value = event['value']
            if isinstance(value, dict):  # 单张图片
                handle_single_image(value)
            elif isinstance(value, str):  # 目录
                handle_multiple_images(value)


if __name__ == '__main__':
    from pywebio.platform.flask import start_server

    start_server(main, port=8080, debug=True)
