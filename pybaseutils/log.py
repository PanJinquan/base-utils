# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2024-10-27 13:42:12
    @Brief  :
"""
import os
import sys
from loguru import logger

LOG_FORMAT = {
    "simple":   "{time:YYYY-MM-DD HH:mm:ss}|{level:7}| {message}",
    "name":     "{time:YYYY-MM-DD HH:mm:ss}|{level:7}|{name} {line:4}| {message}",  # 打印文件名
    "module":   "{time:YYYY-MM-DD HH:mm:ss}|{level:7}|{name}.{module} {line:4}| {message}",  # 打印模块名
    "function": "{time:YYYY-MM-DD HH:mm:ss}|{level:7}|{name}.{module}.{function} {line:4}| {message}",  # 打印函数
}


def set_logger(name="", level="debug", logfile=None, format="simple", is_main_process=True):
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
        logger.configure(handlers=[{"sink": sys.stderr,  # 打印到控制台，sys.stderr表示控制台
                                    "format": format, "colorize": True, "level": level.upper()},
                                   {"sink": logfile,  # 输出到文件，文件名app.log
                                    "format": format, "colorize": False, "level": level.upper(), "rotation": "100 MB",
                                    "retention": "10 days"}
                                   ])
    else:
        logger.configure(handlers=[{"sink": sys.stderr,  # 打印到控制台，sys.stderr表示控制台
                                    "format": format, "colorize": True, "level": "ERROR"}
                                   ])
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
    logger = set_logger(logfile=logfile, is_main_process=True, level="debug")
    example()
