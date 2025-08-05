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
import logging
from logging.handlers import TimedRotatingFileHandler


def get_logger(name=None, level="info", logfile=None, is_main_process=True):
    """
    :param name:
    :param logfile:
    :param level: CRITICAL,FATAL,ERROR,WARN,WARNING,INFO,DEBUG,NOTSET
    :return:
    """
    # 创建logger对象
    if is_main_process:
        logger = logging.getLogger(name)
        # 如果logger已经有，则不添加
        if logger.handlers: return logger
        logger.setLevel(level.upper())
        # 创建格式化器
        formatter = logging.Formatter(
            fmt="%(asctime)s|%(levelname)6s|%(filename)20s|%(funcName)20s|Line%(lineno)4s|%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 创建日志目录
        if logfile:
            os.makedirs(os.path.dirname(logfile), exist_ok=True)
            # 创建文件处理器（按时间轮转）
            file_handler = TimedRotatingFileHandler(
                logfile,
                when='midnight',  # 每天午夜切换新文件
                interval=1,  # 间隔为1天
                backupCount=30,  # 保留30天的日志文件
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    else:
        logger = logging.getLogger(name)
        logger.setLevel("warning")
    return logger


if __name__ == '__main__':
    logger = get_logger(name='APPLOG', level="debug", logfile="./log.log")
    # 测试日志输出
    logger.debug('-----------------')
    logger.debug('这是一条调试信息')
    logger.info('这是一条信息')
    logger.warning('这是一条警告')
    logger.error('这是一条错误信息')
    logger.critical('这是一条严重错误信息')
