# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail :
# @Date   : 2022-04-29 09:13:09
# @Brief  :
# --------------------------------------------------------
"""
import os
import sys
from loguru import logger

LOG_FORMAT = {
    "simple":   "<level>{time:YYYY-MM-DD HH:mm:ss}|{level:7}| {message}</level>",
    "name":     "<level>{time:YYYY-MM-DD HH:mm:ss}|{level:7}|{name} {line:4}| {message}</level>",  # 打印文件名
    "module":   "<level>{time:YYYY-MM-DD HH:mm:ss}|{level:7}|{module} {line:4}| {message}</level>",  # 打印模块名
    # "function": "<level>{time:YYYY-MM-DD HH:mm:ss}|{level:7}|{function} {line:4}| {message}</level>",  # 打印函数
    "function": "<level>{time:YYYY-MM-DD HH:mm:ss}|{level:7}|{module}.{function} {line:4}| {message}</level>",  # 打印函数
    "all":      "<level>{time:YYYY-MM-DD HH:mm:ss}|{level:7}|{name}.{module}.{function} {line:4}| {message}</level>",  # 打印函数
}


def set_logger(name=None, level="debug", logfile=None, format="simple", is_main_process=True):
    """
    logger = set_logger(level="debug", logfile="log.txt")
    url: https://www.cnblogs.com/shiyitongxue/p/17870527.html
    :param level: 设置log输出级别:debug,info,warning,error
    :param logfile: log保存路径，如果为None，则在控制台打印log
    :param is_main_process: 是否是主进程
    :return:
    """
    format = LOG_FORMAT.get(format, LOG_FORMAT.get("simple"))
    if is_main_process:
        # 打印到控制台，sys.stderr表示控制台
        h1 = {"sink": sys.stderr,
              "format": format, "colorize": True, "level": level.upper()}
        # 输出到文件，文件名app.log
        h2 = {"sink": logfile,
              "format": format, "colorize": False, "level": level.upper(), "rotation": "100 MB",
              "retention": "10 days"} if logfile else None
        handlers = [h for h in [h1, h2] if h]
        logger.configure(handlers=handlers)
    else:
        logger.configure(handlers=[{"sink": sys.stderr,  # 打印到控制台，sys.stderr表示控制台
                                    "format": format, "colorize": True, "level": "ERROR"}
                                   ]
                         )
    return logger


def set_logger_v2(name="", level="debug", logfile=None, format=None, is_main_process=True):
    """
    logger = set_logger(level="debug", logfile="log.txt")
    url: https://www.cnblogs.com/shiyitongxue/p/17870527.html
    :param level: 设置log输出级别:debug,info,warning,error
    :param logfile: log保存路径，如果为None，则在控制台打印log
    :param is_main_process: 是否是主进程
    :return:
    """
    format = LOG_FORMAT.get(format, LOG_FORMAT.get("line"))
    logger.remove(0)  # 去除默认的LOG
    if is_main_process:
        # 每天创建一个新的文件，一个星期定期清理一次
        logger.add(logfile, level=level.upper(), rotation="1 day", retention="7 days", format=format)
        logger.add(sys.stderr, level=level.upper(), format=format)
    else:
        logger.add(sys.stderr, level="ERROR", format=format)
    return logger


def get_logger():
    return logger


def example():
    logger = get_logger()
    logger.debug("debug")
    logger.info("info")
    logger.warning("warning")
    logger.error("error")


if __name__ == '__main__':
    logfile = "./log.log"
    # logger = set_logger(logfile=logfile, is_main_process=True, format="function",level="debug")
    logger = set_logger(name="demo", is_main_process=True, format="function", level="debug")
    # logger = set_logger(name="demo", is_main_process=True, format="module", level="debug")
    # logger = get_logger()
    example()
