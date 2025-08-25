# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : 2025-08-25 17:52:03
# @Brief  :
# --------------------------------------------------------
"""
import tornado.ioloop
import tornado.web
import tornado.websocket
import asyncio
from tornado.options import define, options

define("port", default=8888, help="运行端口", type=int)

clients = {}  # 改为字典存储，key为cid，value为连接对象
clients_id = 0  # 用于生成唯一ID
clients_data = []  # 用户数据


class IndexHandler(tornado.web.RequestHandler):
    async def get(self):
        await self.render("index.html")  # 渲染并返回页面


class ChatWebSocketHandler(tornado.websocket.WebSocketHandler):
    def open(self):
        global clients_id
        clients_id += 1
        self.cid = f"用户{clients_id}"
        clients[self.cid] = self  # 用ID作为key存储连接

        print(f"客户端建立连接(ID:{self.cid}),当前连接数: {len(clients)}")
        self.write_message(f"系统: 欢迎使用，你的ID是{self.cid}")
        for msg in clients_data:  # 新用户连接时能看到之前的聊天记录
            self.write_message(msg)

    def on_message(self, message):
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


def application():
    return tornado.web.Application([
        (r"/", IndexHandler),
        (r"/ws", ChatWebSocketHandler),
    ],
        template_path="templates",  # 你的HTML模板目录
        debug=True)


async def main():
    app = application()
    app.listen(options.port)
    print(f"服务器启动在 http://localhost:{options.port}")
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
