# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan (Optimized with APIRouter)
# @E-mail :
# @Date   : 2026-01-26
# @Brief  : 使用APIRouter的优化版图像服务，修复了路由、URL生成错误，并增加了路径遍历防护。
# --------------------------------------------------------
"""
import os
from pathlib import Path
import socket
from fastapi import FastAPI, HTTPException
from fastapi import APIRouter, Header
from fastapi.responses import FileResponse
from pybaseutils import file_utils
router = APIRouter( prefix="/api",tags=["interface"])

app = FastAPI()
port = 40005
host = socket.gethostname()
ip = socket.gethostbyname(host)
root = "images"
ROOT_PATH = Path(root).resolve()  # 使用绝对路径作为基准，防止路径遍历

# --- 关键：定义并配置APIRouter ---
router = APIRouter(prefix="/api/v1", tags=["interface"])

print(f"host: {host}")
print(f"ip  : {ip}")


@router.get("/images/{filename:path}")
async def get_image(filename: str):
    """
    安全地获取 images 目录下的图像文件。
    使用 :path 转换器支持子目录，并通过 resolve() 和 relative_to() 防止路径遍历攻击。
    """
    # 构建目标文件的绝对路径
    target_path = (ROOT_PATH / filename).resolve()

    # 安全检查：确保目标路径在 ROOT_PATH 目录内
    try:
        target_path.relative_to(ROOT_PATH)
    except ValueError:
        raise HTTPException(status_code=403, detail="Forbidden")

    # 检查文件是否存在且为文件（而非目录）
    if target_path.is_file():
        return FileResponse(target_path)
    else:
        raise HTTPException(status_code=404, detail="File not found")


def get_assets_images_url():
    """
    获取所有图片资产的可访问URL列表。
    URL路径与 router 的 prefix 和接口路径保持一致。
    """
    if not ROOT_PATH.exists():
        return []
    files = file_utils.get_files_lists(str(ROOT_PATH), postfix=file_utils.IMG_POSTFIX, sub=True)
    # 修正URL路径，使其与 /api/images/... 路由匹配
    urls = [f"http://{ip}:{port}/api/images/{f}" for f in files]
    return urls


# --- 关键：将 router 注册到主应用 app ---
# 这是使用 APIRouter 的必要步骤，否则定义的路由不会生效 [[1]].
app.include_router(router)

# 主程序入口
if __name__ == "__main__":
    # 确保 images 目录存在
    ROOT_PATH.mkdir(exist_ok=True)

    urls = get_assets_images_url()
    if urls:
        print("Available image URLs:")
        for url in urls:
            print(url)
    else:
        print(f"No images found in '{root}' directory.")

    import uvicorn

    # 启动服务器，监听所有接口
    uvicorn.run(app, host="0.0.0.0", port=port)