# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : 2026-01-21 16:47:36
# @Brief  :
# --------------------------------------------------------
"""
import os
import socket
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pybaseutils import file_utils, image_utils
from fastapi import APIRouter, Header

app = FastAPI()
port = 40001
host = socket.gethostname()
ip = socket.gethostbyname(host)
root = "images"
router = APIRouter(
    prefix="/api",
    tags=["interface"],
)
router.mount("/static", StaticFiles(directory=root), name="static")



print(f"host: {host}")
print(f"ip  : {ip}")


@router.get("/images/{filename:path}")  # 使用:path转换器允许包含斜杠
async def get_image(filename: str):
    path = f"{root}/{filename}"
    print(f"get file: {path}")
    if os.path.exists(path):
        return FileResponse(path)
    return {"error": "File not found"}


def get_assets_images_url():
    file = file_utils.get_files_lists(root, postfix=file_utils.IMG_POSTFIX, sub=True)
    urls = [f"http://{ip}:{port}/{root}/{f}" for f in file]
    return urls


if __name__ == "__main__":
    urls = get_assets_images_url()
    [print(url) for url in urls]
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=port)
