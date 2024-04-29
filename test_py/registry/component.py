# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2023-08-14 13:51:06
    @Brief  :
"""
import logging
from typing import List, Tuple, Dict
from pybaseutils import image_utils, json_utils, numpy_utils
from base import register
from test_py.registry.base import base_instance
from base import Base

logger = logging.getLogger(__name__)


class Component3(Base):
    def __init__(self):
        pass

    def component3_1(self, frame_info, key="", vis=True):
        print(f'key={key},', frame_info)
        return frame_info

    def component3_2(self, frame_info, key="", vis=True):
        print(f'key={key},', frame_info)
        return frame_info


class Component(Base):
    """注册所有组件"""

    def __init__(self):
        # TODO enable_xxx表示开启xxx组件，返回结果component必须用xxx字段表示对应的组件结果
        # TODO 比如：enable_head_helmet表示开启【安全帽】检测，返回结果component中必须使用字段head_helmet表示组件结果
        com = Component3()
        register.put_module(com.component3_1)
        register.put_module(com.component3_2)

        self.modules: dict = register.modules()
        for modules, fun in self.modules.items():
            print(modules, fun)
        self.component = {k: "" for k in self.modules}

    def check_component(self, frame_info):
        """
        TODO enable_xxx表示开启xxx组件，返回结果component必须用xxx字段表示对应的组件结果
        比如：enable_head_helmet表示开启【安全帽】检测，返回结果component中必须使用字段head_helmet表示组件结果
        :param frame_info:
        :return:
        """
        # TODO 判断config中的开启的组件是否存在(注册)
        config: dict = json_utils.get_value(frame_info, ["config"], default={})
        for k in config.keys():
            try:
                v = self.component[k]
            except:
                msg = f"Error:开启组件【{k}】失败，请检查组件名称"
                logger.error(msg)
                frame_info.update({"code": 3002, "msg": msg})

        # TODO 判断返回结果component中的组件字段(key)是否合理
        component: dict = json_utils.get_value(frame_info, ['component'], default={})
        for r in component.keys():
            k = f"enable_{r}"
            title = json_utils.get_value(component, [r, 'title'], default={})
            try:
                self.component[k]
            except:
                msg = f"Error:组件【{title}】返回字段有误，请检查component字段:{r}"
                logger.error(msg)
                frame_info.update({"code": 3003, "msg": msg})
        return frame_info


if __name__ == "__main__":
    config = {
        "image": 1,
        "enable_component_get_tools": 1,
        "enable_workbench_withouttools": 0,
        "enable_boxdoor_closed": 1,
        "enable_boxdoor_closing": 1,
    }
    frame_info = {'image': "image", "metadata": {}, "config": config}
    component = Component()
