# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2023-09-07 09:11:05
    @Brief  :
"""
import traceback
from base import register
from base import register
from base import register
from base import register
from base import Base
from test_py.registry.base import base_instance



def task_component(frame_info, config: dict):
    for key, enable in config.items():
        try:
            if enable: frame_info = register.modules()[key](base_instance, frame_info=frame_info, key=key)
        except:
            traceback.print_exc()
    return frame_info


class Component3(Base):
    def __init__(self):
        pass

    def component3_1(self, frame_info, key="", vis=True):
        print(f'key={key},', frame_info)
        return frame_info

    def component3_2(self, frame_info, key="", vis=True):
        print(f'key={key},', frame_info)
        return frame_info

    def push_component(self):
        # register.put_module(self.component3_1)
        # register.put_module(self.component3_2)
        register.put_module(self.component3_1)
        register.put_module(self.component3_2)

if __name__ == "__main__":
    """
    'component3_2': <function Component3.component3_2 at 0x7fa0605efe50>
    """
    config = {
        # "enable_component_get_tools": 1,
        # "enable_workbench_withouttools": 0,
        # "enable_boxdoor_closed": 1,
        "enable_boxdoor_closing": 1,
        "component3_1": 1,
        "component3_2": 1,
    }
    com = Component3()
    com.push_component()
    print(register.modules())
    frame_info = {'image': "image", "metadata": {}, "config": config}
    frame_info = task_component(frame_info, config)
    print(frame_info)
