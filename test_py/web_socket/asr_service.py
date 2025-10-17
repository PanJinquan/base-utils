# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2025-08-25 17:52:03
# @Brief  :
# --------------------------------------------------------
"""
import sys
import os
import time
import tornado
import tornado.ioloop
import tornado.web
import tornado.websocket
import asyncio
from funasr import AutoModel

expires = 600  # 60秒不活跃则删除
clients = {}  # 改为字典存储，key为cid，value为连接对象
clients_id = 0  # 用于生成唯一ID
clients_data = []  # 用户数据

# 初始化Paraformer流式模型
# model = AutoModel(model="paraformer-zh-streaming", model_revision="v2.0.4")
model_file = "/home/PKing/nasdata/Project/ASR/model/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
model = AutoModel(model=model_file)
cache = {}
chunk_size = [0, 10, 5]  # [0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 4  # number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 1  # number of encoder chunks to lookback for decoder cross-attention


async def check_inactive_clients():
    while True:
        await asyncio.sleep(expires)  # 每分钟检查一次
        t = time.time()
        for cid in list(clients.keys()):
            if t - clients[cid].time > expires:  # 超过60秒未活动
                try:
                    clients[cid].write_message("系统: 你的ID因长时间未活动已被删除")
                    clients[cid].close()
                except:
                    print(f"无法通知{cid}用户")
                del clients[cid]
                print(f"清理不活跃用户(ID:{cid})")


class MainSocketHandler(tornado.websocket.WebSocketHandler):
    def __init__(self, application, request, **kwargs):
        super().__init__(application, request, **kwargs)
        self.time = None
        self.cid = None

    def open(self):
        global clients_id
        clients_id += 1
        self.cid = f"用户{clients_id}"
        self.time = time.time()  # 记录最后活跃时间
        clients[self.cid] = self
        print(f"客户端建立连接(ID:{self.cid}),当前连接数: {len(clients)}")
        self.write_message(f"系统: 欢迎使用，你的ID是{self.cid}")

    def on_message(self, data):
        self.time = time.time()  # 更新最后活跃时间
        print(f"收到来自{self.cid}的消息,data={data[:10]}")
        # 使用FunASR进行流式识别
        result = model.generate(input=data, cache=cache, is_final=False, chunk_size=chunk_size,
                                encoder_chunk_look_back=encoder_chunk_look_back,
                                decoder_chunk_look_back=decoder_chunk_look_back)

        # 返回中间识别结果
        if result and 'text' in result[0]:
            if result[0]['text']: self.write_message(result[0]['text'])

    def on_close(self):
        if self.cid in clients:
            del clients[self.cid]
        print(f"客户端断开连接(ID:{self.cid}),当前连接数: {len(clients)}")

    def check_origin(self, origin):
        return True  # 允许跨域


def web_app():
    handlers = [(r"/asr", MainSocketHandler)]
    app = tornado.web.Application(handlers, debug=True)
    return app


async def main():
    app = web_app()
    tornado.options.define("port", default=8765, help="运行端口", type=int)
    app.listen(tornado.options.options.port)
    print(f"服务器启动在 http://localhost:{tornado.options.options.port}")
    asyncio.create_task(check_inactive_clients())  # 将任务启动移到这里
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
