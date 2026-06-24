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
    "simple": "<level>{time:YYYY-MM-DD HH:mm:ss}|{level:7}| {message}</level>",
    "name": "<level>{time:YYYY-MM-DD HH:mm:ss}|{level:7}|{name} {line}| {message}</level>",  # 打印文件名
    "module": "<level>{time:YYYY-MM-DD HH:mm:ss}|{level:7}|{module} {line}| {message}</level>",  # 打印模块名
    # "function": "<level>{time:YYYY-MM-DD HH:mm:ss}|{level:7}|{function} {line}| {message}</level>",  # 打印函数
    "function": "<level>{time:YYYY-MM-DD HH:mm:ss}|{level:7}|{module}.{function} {line}| {message}</level>",  # 打印函数
    "all": "<level>{time:YYYY-MM-DD HH:mm:ss}|{level:7}|{name}.{module}.{function} {line}| {message}</level>",  # 打印函数
}


def set_logger(name=None, level="debug", logfile=None, format="simple", is_main_process=True,
               rotation="1 days", retention="3 days"):
    """
    logger = set_logger(level="debug", logfile="log.txt")
    url: https://www.cnblogs.com/shiyitongxue/p/17870527.html
    :param level: 设置log输出级别:debug,info,warning,error
    :param logfile: log保存路径，如果为None，则在控制台打印log
    :param is_main_process: 是否是主进程
    :param rotation: 日志文件轮转策略,时间或者大小，默认1天
                    rotation="1 minutes"  # 每分钟轮转
                    rotation="1 hours"    # 每小时轮转
                    rotation="1 days"     # 每天轮转
                    rotation="1 weeks"    # 每周轮转
                    rotation="10 MB"      # 文件达到10MB时轮转
                    rotation="500 MB"     # 文件达到500MB时轮转
    :param retention: 日志文件保留时长，默认3天
                    retention="7 days"    # 保留7天
                    retention="1 months"   # 保留1个月
                    retention="20 GB"     # 保留最近20GB的日志（基于大小）
    :return:
    """
    format = LOG_FORMAT.get(format, LOG_FORMAT.get("line"))
    logger.remove()  # 去除默认的LOG，避免重复打印
    if is_main_process:
        # 每天创建一个新的文件，一个星期定期清理一次
        if logfile: logger.add(logfile, level=level.upper(), rotation=rotation, retention=retention, format=format,
                               enqueue=True,  # 异步写入，会重新打开文件
                               watch=True,  # 避免误删日志文件
                               catch=True,
                               )
        logger.add(sys.stderr, level=level.upper(), format=format)
    else:
        logger.add(sys.stderr, level="ERROR", format=format)
    return logger


def set_logger_v2(name=None, level="debug", logfile=None, format="simple", is_main_process=True,
                  rotation="1 day", retention="3 days"):
    """
    logger = set_logger(level="debug", logfile="log.txt")
    url: https://www.cnblogs.com/shiyitongxue/p/17870527.html
    :param level: 设置log输出级别:debug,info,warning,error
    :param logfile: log保存路径，如果为None，则在控制台打印log
    :param is_main_process: 是否是主进程
    :param rotation: 每rotation，创建新的日志文件
    :param retention: 日志文件保留时长，默认3天
    :return:
    """
    format = LOG_FORMAT.get(format, LOG_FORMAT.get("simple"))
    if is_main_process:
        # 打印到控制台，sys.stderr表示控制台
        h1 = {"sink": sys.stderr,
              "format": format, "colorize": True, "level": level.upper()}
        # 输出到文件，文件名app.log
        h2 = {"sink": logfile,
              "format": format, "colorize": False, "level": level.upper(), "rotation": rotation,
              "retention": retention} if logfile else None
        handlers = [h for h in [h1, h2] if h]
        logger.configure(handlers=handlers)
    else:
        logger.configure(handlers=[{"sink": sys.stderr,  # 打印到控制台，sys.stderr表示控制台
                                    "format": format, "colorize": True, "level": "ERROR"}
                                   ]
                         )
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
    import traceback
    import time

    logfile = "./log.log"
    logger = set_logger(name="demo", is_main_process=True, format="function", level="debug", logfile=logfile)
    for i in range(1000):
        try:
            a = 100 / 0
            logger.debug(f"debug {i}")
            logger.info(f"info {i}")
            logger.warning(f"warning {i}")
            logger.error(f"error {i}")
            time.sleep(1)
        except Exception as e:
            e = traceback.format_exc()
            # logger.error(e)

    # logger = set_logger(logfile=logfile, is_main_process=True, format="function",level="debug")
    # logger = set_logger(name="demo", is_main_process=True, format="module", level="debug")
    # logger = get_logger()
    # example()
