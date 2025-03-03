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
from pywebio.output import *
from pywebio.input import file_upload
from pywebio.session import set_env, defer_call, run_async
from pywebio import start_server
import cv2
import numpy as np

current_images = []  # 存储处理后的图像列表
current_index = 0  # 当前显示索引


def apply_blur(image_bytes):
    """OpenCV高斯模糊处理"""
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    _, encoded = cv2.imencode('.jpg', blurred)
    return encoded.tobytes()


def handle_single_image():
    global current_images, current_index
    file = file_upload("选择图片", accept="image/*")
    if file:
        processed = apply_blur(file['content'])
        current_images = [processed]
        current_index = 0
        with use_scope('main', clear=True):
            put_row([
                put_image(file['content'], width="45%", title="原始图片"),
                put_image(processed, width="45%", title="模糊处理")
            ])


def handle_batch_images():
    global current_images, current_index
    files = file_upload("选择目录", accept="image/*", multiple=True)
    if files:
        current_images = [apply_blur(f['content']) for f in files]
        current_index = 0
        update_display()


def update_display(delta=0):
    global current_index
    if not current_images:
        return
    current_index = (current_index + delta) % len(current_images)
    with use_scope('main', clear=True):
        put_image(current_images[current_index], width="90%",
                  title=f"已处理图片 {current_index + 1}/{len(current_images)}")


# @defer_call
# def cleanup():
#     current_images.clear()


def main():
    set_env(title="图像模糊处理工具")

    # 侧边栏布局
    put_row([
        put_button("选择图片", onclick=handle_single_image),
        put_button("选择目录", onclick=handle_batch_images),
        put_buttons(['← 上一张', '下一张 →'],
                    onclick=[lambda: update_display(-1), lambda: update_display(1)])
    ], position=0, scope='sidebar')

    # 主显示区初始化
    with use_scope('main'):
        put_markdown("## 欢迎使用图像模糊处理工具！")


if __name__ == '__main__':
    start_server(main, port=8080, debug=False, auto_open_webbrowser=True)