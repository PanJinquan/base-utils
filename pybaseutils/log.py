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
import datetime
from loguru import logger

LOG_FORMAT = {
    "simple":   "<level>{extra[time]}|{level:7}| {message}</level>",
    "name":     "<level>{extra[time]}|{level:7}|{name} {line}| {message}</level>",  # 打印文件名
    "module":   "<level>{extra[time]}|{level:7}|{module} {line}| {message}</level>",  # 打印模块名
    "function": "<level>{extra[time]}|{level:7}|{module}.{function} {line}| {message}</level>",
    "precise":  "<level>{extra[time]}|{level:7}|{module}.{function} {line}| {message}</level>",
    "all":      "<level>{extra[time]}|{level:7}|{name}.{module}.{function} {line}| {message}</level>",# 打印函数
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

    def call_time(record):
        # 显式固化 logger 调用时刻，避免异步 sink 输出时产生“当前打印时间”的误解
        if format == "precise":
            record["extra"]["time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        else:
            record["extra"]["time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f = LOG_FORMAT.get(format, LOG_FORMAT.get("simple"))
    logger.remove()  # 去除默认的LOG，避免重复打印
    logger_ = logger.patch(call_time)
    if is_main_process:
        if logfile: logger_.add(logfile, level=level.upper(), rotation=rotation, retention=retention, format=f,
                                enqueue=True,  # 异步写入，会重新打开文件
                                watch=True,  # 避免误删日志文件
                                catch=True,
                                )
        logger_.add(sys.stderr, level=level.upper(), format=f)
    else:
        logger_.add(sys.stderr, level="ERROR", format=f)
    return logger_


def get_logger():
    return logger


if __name__ == '__main__':
    import traceback
    import time

    logfile = "./log.log"
    logger = set_logger(name="demo", is_main_process=True, format="precise", level="debug", logfile=logfile)
    # logger = set_logger(name="demo", is_main_process=True, format="module", level="debug", logfile=logfile)

    for i in range(1000):
        try:
            t = time.time()
            date = datetime.datetime.fromtimestamp(t).strftime("%H:%M:%S.%f")[:-3]
            logger.debug(f"{date} debug {i}")
            # logger.info(f"info {i}")
            # logger.warning(f"warning {i}")
            # logger.error(f"error {i}")
            time.sleep(0.01)
        except Exception as e:
            e = traceback.format_exc()
            logger.error(e)
