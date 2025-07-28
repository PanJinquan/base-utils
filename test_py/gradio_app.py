# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2025-05-16 09:04:18
# @Brief  : 教程https://gradio.org.cn/guides/quickstart
# --------------------------------------------------------
"""
import numbers
import os
import cv2
import shutil
import numpy as np
import gradio as gr
from pybaseutils import file_utils, image_utils, time_utils

ptr = -1
cache_file = []
cache_root = ""

# TODO 定义CSS样式
CSS = """
.bordered1 {
    border: 2px solid #007BFF; /* 蓝色边框 */
    padding: 10px;             /* 内边距   */
    margin: 5px;               /* 外边距   */
    border-radius: 5px;        /* 圆角     */
.bordered21 {
    height: 900px;
    width: 50%;
    border: 1px solid #007BFF; /* 蓝色边框 */
    padding: 6px;              /* 内边距   */
    margin: 3px;               /* 外边距   */
    border-radius: 5px;        /* 圆角     */
}
.bordered22 {
    height: 900px;
    width: 50%;
    border: 1px solid #007BFF; /* 蓝色边框 */
    padding: 6px;              /* 内边距   */
    margin: 3px;               /* 外边距   */
    border-radius: 5px;        /* 圆角     */
}
"""


def clip(x, min_, max_): return max(min_, min(x, max_))


def task(image, **kwargs):
    image = cv2.GaussianBlur(image, (15, 15), 0)
    return image


def task_callback(ptr):
    """
    :param ptr: 第N张图片
    :return: log_info: LOG显示信息
             img_info：图片信息[(file,text)]
    """
    log_info, img_info = "", []
    if not cache_file: return log_info, img_info
    ptr = clip(ptr, 0, len(cache_file) - 1)
    path = cache_file[ptr]
    image = cv2.imread(path)
    image = task(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    log_info = "[{}/{}]正在处理,file={}".format(ptr + 1, len(cache_file), path)
    img_info = [(image, path)]
    print(log_info)
    return log_info, img_info


def upload(files):
    global cache_root
    print("上传数据...")
    log_info, img_info = "", []
    if not files: return log_info, img_info
    tmp_tabel = {file: file for file in cache_file}  # 检查是否存在重复的文件
    for file in files:
        try:
            path, name = os.path.split(file.name)
            cache_root = os.path.dirname(path)
            if os.path.exists(file.name) and file.name in tmp_tabel:
                print("have exist:{}".format(file.name))
            else:
                cache_file.append(file.name)
                print("add file  :{}".format(file.name))
        except Exception as e:
            print(e)
    img_info = [cache_file[0]]  # 未避免显示太多，仅展示一张图片
    log_info = "上传{}个文件，总共{}个文件".format(len(files), len(cache_file))
    print(log_info)
    return log_info, img_info


def load_page_images(page=1, page_size=10):
    total_pages = (len(cache_file) + page_size - 1) // page_size
    start_idx = (page - 1) * page_size
    return cache_file[start_idx: start_idx + page_size], total_pages


def task_all():
    print("处理所有数据...")
    img_info = []
    log_info = ""
    for ptr in range(len(cache_file)):
        log_info, img = task_callback(ptr)
        img_info.extend(img)
    return log_info, img_info


def next():
    global ptr
    ptr = clip(ptr + 1, 0, len(cache_file) - 1) if cache_file else -1
    print(f"下一张,ptr={ptr}")
    return task_callback(ptr)


def last():
    global ptr
    ptr = clip(ptr - 1, 0, len(cache_file) - 1) if cache_file else -1
    print(f"上一张,ptr={ptr}")
    return task_callback(ptr)


def clear():
    global ptr, cache_file
    ptr = -1
    cache_file = []
    # if cache_root.endswith("gradio"): file_utils.remove_dir(cache_root)
    print("清空所有数据:{}".format(cache_root))
    return task_callback(ptr)


def create_web_ui(title="图片搜索系统"):
    with gr.Blocks(title=title, css=CSS) as demo:
        gr.Markdown(f"<center><h1>{title}</h1></center>")  # 使用HTML标签
        with gr.Row(elem_classes="bordered1"):  # 水平布局分为左右两列
            with gr.Column(scale=1, min_width=10, elem_classes="bordered21"):  # 左侧边栏
                btn11 = gr.Button("提交数据", variant="primary")
                with gr.Row():  # 水平布局分为左右两列
                    btn31 = gr.Button("上一张", variant="primary", min_width=1)
                    btn32 = gr.Button("下一张", variant="primary", min_width=1)
                btn21 = gr.Button("全部处理", variant="primary")
                btn41 = gr.Button("清空数据", variant="primary")
            with gr.Column(scale=8, elem_classes="bordered22"):  # 主内容区
                inp1 = gr.File(label="选择图片", file_count="multiple", height=150, file_types=["image"],
                               type="filepath")
                # info = gr.Text(value="info", label="info", show_label=False, lines=1)
                info = gr.Markdown(f"info")  # 使用HTML标签
                res1 = gr.Gallery(label="结果显示", show_label=True, height="auto", columns=5)
            outputs = [info, res1]
            btn11.click(upload, inputs=inp1, outputs=outputs)
            btn21.click(task_all, outputs=outputs)
            btn41.click(clear, outputs=outputs)
            btn31.click(last, outputs=outputs)
            btn32.click(next, outputs=outputs)
    return demo


if __name__ == "__main__":
    app = create_web_ui()
    app.launch(share=True)
