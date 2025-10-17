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

expires = 60  # 60秒不活跃则删除
clients = {}  # 改为字典存储，key为cid，value为连接对象
clients_id = 0  # 用于生成唯一ID
clients_data = []  # 用户数据


class IndexHandler(tornado.web.RequestHandler):
    async def get(self):
        await self.render("index.html")  # 渲染并返回聊天室页面


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
        for msg in clients_data:  # 新用户连接时能看到之前的聊天记录
            self.write_message(msg)

    def on_message(self, message):
        self.time = time.time()  # 更新最后活跃时间
        print(f"收到来自{self.cid}的消息: {message}")
        formatted_msg = f"{self.cid}: {message}"  # 使用分配的ID
        clients_data.append(formatted_msg)
        for client_id, client in clients.items():
            try:
                client.write_message(formatted_msg)
            except:
                print(f"向{client_id}发送消息失败")

    def on_close(self):
        if self.cid in clients:
            del clients[self.cid]
        print(f"客户端断开连接(ID:{self.cid}),当前连接数: {len(clients)}")

    def check_origin(self, origin):
        return True  # 允许跨域


def web_app():
    handlers = [
        (r"/", IndexHandler),
        (r"/ws", MainSocketHandler)
    ]
    app = tornado.web.Application(handlers,
                                  template_path="templates",  # 你的HTML模板目录
                                  debug=True)
    return app


async def main():
    app = web_app()
    tornado.options.define("port", default=8888, help="运行端口", type=int)
    app.listen(tornado.options.options.port)
    print(f"服务器启动在 http://localhost:{tornado.options.options.port}")
    asyncio.create_task(check_inactive_clients())  # 将任务启动移到这里
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
