# -*-coding: utf-8 -*-
"""
    @Author : PanJinquan
    @E-mail : 
    @Date   : 2025-02-17 10:17:35
    @Brief  :
"""
import cv2
import os
import streamlit as st
import numpy as np
from typing import List, Tuple, Callable
from PIL import Image


def set_page_config(page_title="主页", layout="wide"):
    # 设置页面显示配置
    st.set_page_config(page_title=page_title, layout=layout)  # 使用宽屏模式
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
        st.image(image, channels="BGR")
        # 处理图片
        result = {}
        if callback: result = callback(image, **kwargs)
        # 显示处理后的图片
        st.subheader("处理结果")
        st.text(result.get("info"))
        if isinstance(result.get("image", None), np.ndarray): st.image(image, channels="BGR")


def st_floder_image(callback: Callable = None, **kwargs):
    # 上传文件夹
    uploaded_files = st.sidebar.file_uploader("选择图片", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    if uploaded_files:
        # 创建图片索引
        if 'image_index' not in st.session_state:
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
        image = cv2.cvtColor(np.array(Image.open(uploaded_files[st.session_state.image_index])), cv2.COLOR_RGB2BGR)
        # 显示原图
        st.subheader(f"原始图像 ({st.session_state.image_index + 1}/{len(uploaded_files)})")
        st.image(image, channels="BGR")
        # 处理图片
        result = {}
        if callback: result = callback(image, **kwargs)
        # 显示处理后的图片
        st.subheader("处理结果")
        st.text(result.get("info"))
        if isinstance(result.get("image", None), np.ndarray): st.image(image, channels="BGR")


def st_page_setup(title="图像处理工具", callback: Callable = None, **kwargs):
    """
    :param title:
    :param callback: 回调函数(image, **kwargs)
    :param kwargs:
    :return:
    """
    st.title(title)
    # set_page_config(page_title=title, layout="wide")
    # 侧边栏
    st.sidebar.title("选择图片")
    # 选择处理模式
    mode = st.sidebar.radio("选择模式", ["单张图片处理", "批量图片处理"])
    if mode == "单张图片处理":
        st_single_image(callback=callback, **kwargs)
    else:
        st_floder_image(callback=callback, **kwargs)


class Task(object):
    def __init__(self):
        print("初始化")

    def task(self, image, **kwargs):
        """
        :param image: BGR image
        :param kwargs:
        :return:
        """
        kernel_size = (15, 15)
        return cv2.GaussianBlur(image, kernel_size, 0)


if __name__ == "__main__":
    """streamlit run web.py"""
    # 使用session_state来存储Task实例，确保只初始化一次
    if 'task_instance' not in st.session_state:
        st.session_state.task_instance = Task()

    st_page_setup(callback=st.session_state.task_instance.task)
