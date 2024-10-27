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


def set_logger(name="", level="debug", logfile=None, format=None, is_main_process=True):
    """
    logger = set_logger(level="debug", logfile="log.txt")
    url: https://www.cnblogs.com/shiyitongxue/p/17870527.html
    :param level: 设置log输出级别:debug,info,warning,error
    :param logfile: log保存路径，如果为None，则在控制台打印log
    :param is_main_process: 是否是主进程
    :return:
    """
    if not format: format = "{time:YYYY-MM-DD HH:mm:ss}|{level:7s}|{message}"
    logger.remove(0)  # 去除默认的LOG
    if is_main_process:
        logger.add(logfile, level=level.upper(), rotation="1 day", retention="7 days", format=format)
        logger.add(sys.stderr, level=level.upper(), format=format)
    else:
        logger.add(sys.stderr, level="ERROR", format=format)
    return logger


def get_logger():
    return logger


if __name__ == '__main__':
    logfile = "./log.log"
    logger = set_logger(logfile=logfile, is_main_process=False, level="debug")
    logger = get_logger()
    logger.debug("debug")
    logger.info("info")
    logger.warning("warning")
    logger.error("error")
