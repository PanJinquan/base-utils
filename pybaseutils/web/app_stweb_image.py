# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2025-02-18 16:10:49
# @Brief  : 基于streamlit实现图像处理web小工具
#           运行方法：streamlit run app_stweb_image.py
# --------------------------------------------------------
"""

import cv2
import os
import streamlit as st
import numpy as np
from typing import List, Tuple, Callable
from PIL import Image


def set_page_config(page_title="主页", layout="wide"):
    """
    :param page_title:
    :param layout: wide,centered
    :return:
    """
    # 设置页面显示配置
    if layout: st.set_page_config(page_title=page_title, layout=layout)  # 使用宽屏模式
    # 创建Streamlit应用标题
    st.title(page_title)


def st_single_image(callback: Callable = None, **kwargs):
    # 上传单张图片
    file = st.sidebar.file_uploader("选择图片", type=['png', 'jpg', 'jpeg'])
    if file is not None:
        print(f"do {file}")
        # 转换为OpenCV格式
        image = cv2.cvtColor(np.array(Image.open(file)), cv2.COLOR_RGB2BGR)
        # 显示原图
        st.subheader("原始图像")
        st.image(image, channels="BGR", use_container_width=True, caption="原始图像")
        # 处理图片
        result = {}
        if callback: result = callback(image, **kwargs)
        # 显示处理后的图片
        st.subheader("处理结果")
        st.text(result.get("info"))
        if isinstance(result.get("image", None), np.ndarray):
            st.image(image, channels="BGR", use_container_width=True, caption="处理结果")


def st_floder_image(callback: Callable = None, **kwargs):
    # 上传文件夹
    uploaded_files = st.sidebar.file_uploader("选择图片", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    if uploaded_files:
        # 检查是否有文件被上传
        if len(uploaded_files) == 0:
            st.warning("请上传图片文件")
            return

        # 确保索引在有效范围内
        if 'image_index' not in st.session_state or st.session_state.image_index >= len(uploaded_files):
            st.session_state.image_index = 0

        # 添加导航按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("上一张") and st.session_state.image_index > 0:
                st.session_state.image_index -= 1
        with col2:
            if st.button("下一张") and st.session_state.image_index < len(uploaded_files) - 1:
                st.session_state.image_index += 1

        # 显示当前图片
        file = uploaded_files[st.session_state.image_index]
        print(f"do {file}")
        image = cv2.cvtColor(np.array(Image.open(file)), cv2.COLOR_RGB2BGR)
        # 显示原图
        st.subheader(f"原始图像 ({st.session_state.image_index + 1}/{len(uploaded_files)})")
        st.image(image, channels="BGR", use_container_width=True, caption="原始图像")
        # 处理图片
        result = {}
        if callback: result = callback(image, **kwargs)
        # 显示处理后的图片
        st.subheader("处理结果")
        st.text(result.get("info"))
        if isinstance(result.get("image", None), np.ndarray):
            st.image(image, channels="BGR", use_container_width=True, caption="处理结果")


def st_page_setup(title="图像处理工具", layout="", callback: Callable = None, **kwargs):
    """
    streamlit run web.py
    :param title:
    :param callback: 回调函数(image, **kwargs)
    :param kwargs:
    :return:
    """
    set_page_config(page_title=title, layout=layout)
    # 侧边栏
    st.sidebar.title("选择图片")
    # 选择处理模式
    mode = st.sidebar.radio("选择模式", ["单张图片处理", "批量图片处理"])
    if mode == "单张图片处理":
        st_single_image(callback=callback, **kwargs)
    else:
        st_floder_image(callback=callback, **kwargs)


class Example(object):
    def __init__(self):
        print("初始化")

    def task(self, image, **kwargs):
        """
        :param image: BGR image
        :param kwargs:
        :return:
        """
        kernel_size = (15, 15)
        image = cv2.GaussianBlur(image, kernel_size, 0)
        return dict(image=image, info="高斯模糊")


if __name__ == "__main__":
    # 使用session_state来存储Task实例，确保只初始化一次
    if 'task_instance' not in st.session_state:
        st.session_state.task_instance = Example()

    st_page_setup(callback=st.session_state.task_instance.task)
