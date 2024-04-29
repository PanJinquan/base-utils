# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2023-09-07 09:10:34
    @Brief  :
"""
from test_py.registry.register import register



class Base(object):
    def __init__(self):
        pass


class Component1(Base):
    def __init__(self):
        pass

    @register.put_module()
    def enable_boxdoor_closed(self, frame_info, key="", vis=True):
        print(f'key={key},', frame_info)
        return frame_info

    @register.put_module()
    def enable_boxdoor_closing(self, frame_info, key="", vis=True):
        print(f'key={key},', frame_info)
        return frame_info

    @register.put_module()
    def test_boxdoor_closing(self, frame_info, key="", vis=True):
        print(f'key={key},', frame_info)
        return frame_info


class Component2(Base):
    def __init__(self):
        pass

    @register.put_module()
    def enable_component_get_tools(self, frame_info, key="", vis=True):
        print(f'key={key},', frame_info)
        return frame_info

    @register.put_module()
    def enable_workbench_withouttools(self, frame_info, key="", vis=True):
        print(f'key={key},', frame_info)
        return frame_info


base_instance = Base()
