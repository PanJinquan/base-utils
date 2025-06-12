# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2025-05-16 09:04:18
# @Brief  : 教程https://gradio.org.cn/guides/quickstart
# --------------------------------------------------------
"""
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : 2025-05-23 15:40:50
# @Brief  :
# --------------------------------------------------------
"""
import os.path
import gradio as gr
import pandas as pd

file_info = {}  # TODO 文件列表
file_path = None  # TODO 当前处理的文件


def get_image_viewer(path, name):
    return [(path, name)]


def get_pdf_viewer():
    pdf_data = []
    for i, (p, l) in enumerate(file_info.items()):
        pdf_data.append({"标签": l, "图像路径": p})
    pdf_viewer = pd.DataFrame(pdf_data)
    return pdf_viewer


def fn_del_file_list():
    file_info.clear()
    pdf_viewer = get_pdf_viewer()
    txt_viewer = ""  # 更新label
    img_viewer = []
    return pdf_viewer, txt_viewer, img_viewer


def fn_get_file_list(files):
    global file_path
    if files is None: files = []
    if files: file_path = files[0]
    for file in files:
        if file not in file_info: file_info[file] = ""
    pdf_viewer = get_pdf_viewer()
    return pdf_viewer


def fn_set_file_label(name):
    if isinstance(file_path, str) and os.path.exists(file_path):
        file_info[file_path] = name
        img_viewer = get_image_viewer(file_path, name)  # 更新图像
        txt_viewer = name  # 更新label
    else:
        txt_viewer = ""  # 更新label
        img_viewer = []
    pdf_viewer = get_pdf_viewer()  # 更新表格
    return pdf_viewer, txt_viewer, img_viewer


def fn_show_selected_image(evt: gr.SelectData):
    global file_path
    if isinstance(evt.value, str) and os.path.exists(evt.value):
        file_path = evt.value
        name = file_info.get(file_path, "")
        txt_viewer = name
        if not name: name = os.path.basename(file_path)
        img_viewer = get_image_viewer(file_path, name)
    else:
        file_path = ""
        txt_viewer = ""
        img_viewer = []
    return txt_viewer, img_viewer


# TODO 定义CSS样式
CSS = """
.bordered1 {
    border: 2px solid #007BFF; /* 蓝色边框 */
    padding: 10px;             /* 内边距   */
    margin: 5px;               /* 外边距   */
    border-radius: 5px;        /* 圆角     */
.bordered21 {
    height: 100%;
    width: 100%;
    border: 1px solid #007BFF; /* 蓝色边框 */
    padding: 6px;              /* 内边距   */
    margin: 3px;               /* 外边距   */
    border-radius: 5px;        /* 圆角     */
}
.bordered22 {
    height: 100%;
    width: 100%;
    border: 1px solid #007BFF; /* 蓝色边框 */
    padding: 6px;              /* 内边距   */
    margin: 3px;               /* 外边距   */
    border-radius: 5px;        /* 圆角     */
}
"""


def create_ui_tag1(tag="注册"):
    with gr.Tab(label=tag):
        with gr.Row(elem_classes="bordered1"):  # 水平布局分为左右两列
            with gr.Column(scale=1, min_width=10, elem_classes="bordered21"):  # 左侧边栏
                reg_objs = gr.Button("注册目标", variant="primary")
                del_file = gr.Button("清空文件", variant="primary")
                vis_db = gr.Button("显示数据库", variant="primary")
                del_db = gr.Button("删除数据库", variant="primary")
            with gr.Column(scale=8, elem_classes="bordered22"):  # 主内容区
                file_inp = gr.File(label="导入图片", file_count="multiple", file_types=["image"], height=150)
                file_table = gr.Dataframe(label="选择图片", show_label=True, col_count=0,
                                          max_height=200, show_row_numbers=True)
                text_inp = gr.Textbox(label="编辑标签(请点击图像路径编辑标签)", lines=1, show_label=True)
                show_image = gr.Gallery(label="结果显示", show_label=True, height="auto", columns=5)
            file_inp.change(fn=fn_get_file_list, inputs=file_inp, outputs=file_table)
            del_file.click(fn=fn_del_file_list, inputs=None, outputs=[file_table, text_inp, show_image])
            text_inp.change(fn=fn_set_file_label, inputs=text_inp, outputs=[file_table, text_inp, show_image])
            file_table.select(fn=fn_show_selected_image, inputs=None, outputs=[text_inp, show_image])


def create_ui_tag2(tag="检索"):
    with gr.Tab(label=tag):
        with gr.Row(elem_classes="bordered1"):  # 水平布局分为左右两列
            pass


def create_ui(title="图片检索工具"):
    with gr.Blocks(title=title, css=CSS) as demo:
        gr.Markdown(f"<center><h1>{title}</h1></center>")  # 使用HTML标签
        create_ui_tag1()
        create_ui_tag2()
    return demo


if __name__ == "__main__":
    # 启动Gradio应用
    app = create_ui()
    app.launch(share=True)
