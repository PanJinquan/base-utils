import os
import logging
import threading
import sys
import time

local = threading.local()
operateid_key = "operateid"


def set_operate_id(val):
    local.__setattr__(operateid_key, val)


class ThreadLocalFormatter(logging.Formatter):

    def format(self, record):
        msg = super(ThreadLocalFormatter, self).format(record)
        try:
            opid = local.__getattribute__(operateid_key)
            msg = opid + " " + msg
        except Exception:
            pass

        return msg


log_level = os.getenv("LOG_LEVEL", "info").upper()
# LOG_FORMAT = "%(asctime)s %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
logging.Formatter.default_msec_format = '%s.%03d'
logging.Formatter.default_time_format = '%Y-%m-%dT%H:%M:%S'

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(ThreadLocalFormatter(fmt=LOG_FORMAT))
# 直接设置datefmt毫秒时间戳无效 有疑问请查看logging.Formatter代码 不推荐使用handlers配置
# logging.basicConfig(level=log_level, format=LOG_FORMAT, handlers=[stream_handler])
logging.basicConfig(level=log_level, format=LOG_FORMAT)

# 重置handlers
for h in logging.root.handlers:
    logging.root.removeHandler(h)
logging.root.addHandler(stream_handler)

log = logging.getLogger("EP")


def run_time_decorator(title=""):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # torch.cuda.synchronize()
            t0 = time.time()
            result = func(*args, **kwargs)
            # torch.cuda.synchronize()
            t1 = time.time()
            # print("{} call {} elapsed: {}ms ".format(title, func.__name__, (t1 - t0) * 1000))
            log.info("{}\t call {} \t elapsed:{:4.3f}ms \t".format(title, func.__name__, (t1 - t0) * 1000))
            return result

        return wrapper

    return decorator
