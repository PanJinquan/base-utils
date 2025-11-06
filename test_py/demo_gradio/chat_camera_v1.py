# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2025-10-31 10:28:59
# @Brief  : 简单的摄像头播放功能 - 使用gr.Blocks构建
# --------------------------------------------------------
"""
import cv2
import time
import numpy as np
import gradio as gr
from pybaseutils import http_utils, image_utils, file_utils
from pybaseutils.cvutils import video_utils

url = "http://192.168.68.102:40000/api/v1/chat"


class WebState:
    code = 1
    msgs = {-1: "等待", 0: "结束", 1: "开始", 2: "继续", 3: "暂停", 4: "播放"}


state = WebState()


def finish(*args):
    state.code = 0  # 结束状态
    print(f"state={state.msgs[state.code]},input={args}")


def display():
    if state.code == 4:
        state.code = 3  # 如何在播放状态，则切换为暂停状态
    else:
        state.code = 4  # 切换为播放状态
    print(f"state={state.msgs[state.code]}")
    return state.msgs[state.code]


def image_process(src: str | np.ndarray):
    """
    :param src: RGB
    :return:
    """
    if src is None: return None, None
    if isinstance(src, str): src = cv2.imread(src)[:, :, ::-1]
    # print(f"处理时间: {int(time.time())}", src.shape, src.dtype)
    out = cv2.resize(src, (224, 224))
    out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    return src, out


def video_generator(video: int | str):
    """
    :param video: 视频路径或摄像头索引
    :return:
    """
    if isinstance(video, str) and len(video) == 1: video = int(video)
    print(f"state={state.msgs[state.code]},video={video}")
    if video == "" or video is None:
        state.code = -1  # TODO 避免循环播放
        yield None, None
    # w, h, num, fps = video_utils.get_video_info(video, vis=False)
    video_cap = video_utils.video_iterator(video, save_video=None)
    state.code = 1  # TODO 开始处理视频
    print(f"state={state.msgs[state.code]},video={video}")
    for data_info in video_cap:
        while state.code == 3:  # 暂停状态，等待继续状态
            time.sleep(0.05)
        if state.code == 0: break
        src = data_info["frame"]
        src = src[:, :, ::-1]  # BGR to RGB
        src, out = image_process(src)
        state.code = 4  # 播放状态
        yield src, out  # RGB image
    state.code = 0  # TODO 处理完成
    print(f"state={state.msgs[state.code]},video={video}")
    yield None, None


def get_history(history=[]):
    """
    获取历史记录中消息和文件
    :param history: 聊天记录，包含用户输入的文本消息和文件，以及系统返回的文本消息和文件
                    history=[{"role": "user",      "content": "用户输入"},
                             {"role": "assistant", "content": "系统结果"}]
    :return: texts文本列表(不含文件), files文件列表
    """
    files, texts = [], []
    for h in history:
        content = h.get("content", [])
        if content and h.get("role", "") == "user" and isinstance(content, list):
            files.extend(content)
        else:
            texts.append(h)
    return texts, files


def request_process(texts="", image=None, video=None, history=[]):
    """
    处理系统返回的文本消息和文件
    :param result: 系统返回的消息，包含文本消息和文件
    :return: 系统返回的文本消息和文件
    """
    try:
        inputs = {'role': 'user', 'content': texts, 'image': image, 'video': video}
        params = {"reqid": file_utils.get_time(), "data": {"prompts": inputs, "history": history}}
        result = http_utils.post(url, params=params)
        print(result)
        result = result.get("data", {})
    except Exception as e:
        result = {"role": "assistant", "content": "系统错误:接口请求失败,url={}".format(url)}
    return result


def chat_image_process(message: dict, history=[]):
    """
    处理用户输入的文本消息和文件，返回系统的文本消息和文件
    :param message: 用户输入的消息，包含文本消息和文件
    :param history: 聊天记录，包含用户输入的文本消息和文件，以及系统返回的文本消息和文件
                    history=[{"role": "user",      "content": "用户输入"},
                             {"role": "assistant", "content": "系统结果"}]
    :return: 系统返回的文本和文件消息
    """
    # TODO 1.用户输入的文本和文件消息
    his_texts, his_files = get_history(history)
    inp_text = message.get("text", "")
    inp_file = message.get("files", [])  # 仅仅支持一个文件(图片或视频)
    if not inp_text: return "", history
    use_file = inp_file[0] if inp_file else his_files[-1] if his_files else ""
    print("history files ={}".format(his_files))
    print("select  file  ={}".format(use_file))
    # TODO 2.用户显示的文本和文件消息
    if inp_file:
        history.append({"role": "user", "content": inp_file})
    if inp_text:
        history.append({"role": "user", "content": inp_text})
    # TODO 3.系统处理文本和文件消息
    print(f"state={state.msgs[state.code]},use_file={use_file}")
    image, video = None, None
    if file_utils.is_image(use_file):
        image = image_utils.read_image_base64(use_file)
    elif file_utils.is_video(use_file):
        video = None
    result = request_process(texts=inp_text, image=image, video=video, history=his_texts)
    # TODO 5.系统显示的文本和文件消息
    if result:
        history.append(result)
    # TODO 第一个元素为空,用于清空输入框
    return "", history


CSS = """
.web_title {
    text-align: center;
    margin: 5px 0;                /* 上下间距5px,左右0px */
    padding: 0px;                 /* 内边距 */
    color: #000000;
}
.web_footer {                     /* 自定义页脚 */
    text-align: center;           /*           */
    margin-top: 250px;
    color: #666;
}
.gradio-container {
    width: 100% !important;
    max-width: none !important;
}
.gradio-image {
    height: 480px;
}
.gradio-image img {
    height: 480px;
    object-fit: contain !important;
}
.row-align {                 
    display: flex;
    justify-content: center;
    align-items: center;
}
.button-style {
    text-align: center;
    height: 35px;
    max-width: 150px;
    font-size: 15px;
}
"""


def ui_chatbot(name=""):
    with gr.Tab(label=name):
        out = gr.Chatbot(label="聊天记录",
                         height=500,
                         show_copy_button=True,
                         render_markdown=True,
                         sanitize_html=False,
                         avatar_images=["data/assets/user.png", "data/assets/system.png"],
                         type="messages"
                         )
        inp = gr.MultimodalTextbox(interactive=True,
                                   placeholder="输入文本消息，或者上传图片/视频/音频文件",
                                   show_label=False,
                                   file_types=["image", "video", "audio", ".pdf", ".txt"],
                                   stop_btn="停止",
                                   )
        inp.stop(fn=finish)
        inp.submit(fn=chat_image_process,
                   show_progress_on=[out],
                   inputs=[inp, out],
                   outputs=[inp, out]
                   )


def ui_images(name=""):
    with gr.Tab(label=name):
        inp = gr.File(label="图像", file_count="single", file_types=["image"], height=200)
        with gr.Row():
            src = gr.Image(label="原始", elem_classes=["gradio-image"])
            out = gr.Image(label="结果", elem_classes=["gradio-image"])
        inp.change(fn=image_process, inputs=[inp], outputs=[src, out])


def ui_videos(name=""):
    with gr.Tab(label=name):
        inp = gr.File(label="视频", file_count="single", file_types=["video"], height=200)
        with gr.Row(elem_classes="row-align"):
            btn1 = gr.Button("开始", variant="primary", elem_classes="button-style")
            btn2 = gr.Button("结束", variant="primary", elem_classes="button-style")
        with gr.Row():
            src = gr.Image(label="原始", streaming=True, elem_classes=["gradio-image"])
            out = gr.Image(label="结果", streaming=True, elem_classes=["gradio-image"])
        # inp.change(fn=video_generator, inputs=inp, outputs=[src, out], scroll_to_output=False)
        btn1.click(fn=video_generator, inputs=inp, outputs=[src, out], scroll_to_output=False)
        btn1.click(fn=display, outputs=[btn1])
        btn2.click(fn=finish)


def ui_camera(name=""):
    with gr.Tab(label=name):
        inp = gr.Textbox(label="摄像头ID", placeholder="请输入摄像头ID，如0、1、2等")
        with gr.Row(elem_classes="row-align"):
            btn1 = gr.Button("开始", variant="primary", elem_classes="button-style")
            btn2 = gr.Button("结束", variant="primary", elem_classes="button-style")
        with gr.Row():
            src = gr.Image(label="原始", streaming=True, elem_classes=["gradio-image"])
            out = gr.Image(label="结果", streaming=True, elem_classes=["gradio-image"])
        # inp.submit(fn=video_generator, inputs=inp, outputs=[src, out], scroll_to_output=False)
        btn1.click(fn=video_generator, inputs=inp, outputs=[src, out], scroll_to_output=False)
        btn1.click(fn=display, outputs=[btn1])
        btn2.click(fn=finish)


def ui_webcam(name=""):
    with gr.Tab(label=name):
        with gr.Row():
            with gr.Column(scale=2):
                inp = gr.Image(label="摄像头", sources=["webcam"], streaming=True, elem_classes=["gradio-image"])
            with gr.Column(scale=2):
                out = gr.Image(label="结果", elem_classes=["gradio-image"])
            inp.stream(fn=image_process, inputs=[inp], outputs=[out], stream_every=0.030)


def ui_titles(name=""):
    # 自定义标题
    ui = gr.HTML(f""" 
                <h1 class="web_title">{name}</h1>
                """)
    return ui


def ui_footer():
    # 自定义页脚
    ui = gr.HTML("""<div class="web_footer">
                    Created by PanJinquan | Contact: pan_jinquan@163.com
                    </div>
                """)
    return ui


def create_app(title="AI小工具"):
    """
    :param title:
    :return:
    """
    with gr.Blocks(title=title, css=CSS) as app:
        ui_titles(name=title)
        ui_chatbot(name="聊天")
        ui_images(name="图像")
        ui_videos(name="视频")
        # ui_webcam(name="webcam")
        ui_camera(name="摄像头")
        ui_footer()
    return app


if __name__ == "__main__":
    # 创建并启动界面
    app = create_app()
    app.launch(share=False, show_error=True)
