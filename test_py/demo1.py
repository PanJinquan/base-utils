# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import pandas as pd


class AppState:
    def __init__(self, ):
        self.code = 1
        self.db_cl_names = []
        self._cl_name = self.db_cl_names[0] if self.db_cl_names else None

    @property
    def cl_name(self):
        if not self._cl_name:
            self._cl_name = self.db_cl_names[0] if self.db_cl_names else None
        return self._cl_name

    @cl_name.setter
    def cl_name(self, v):
        assert v in self.db_cl_names, f'cl_name {v} not in db_cl_names {self.db_cl_names}'

        self._cl_name = v


def is_sorted_numpy(arr: np.ndarray) -> bool:
    """
    使用 numpy 判断数组是否为升序。
    对于大型数值数组，这是最快的方法。
    注意：此函数输入应为 numpy 数组。
    """
    # np.diff 计算相邻元素之差，如果升序，则所有差值都 >= 0
    # np.all() 检查所有条件是否为真
    return np.all(np.diff(arr) >= 0)


if __name__ == '__main__':
    import numpy as np

    # --- 示例 ---
    # 假设 data 是一个 numpy 数组
    np_data = np.array([1, 2, 2, 3, 5, 8, 0])
    result = is_sorted_numpy(np_data)
    print(f"Numpy 数组 {np_data} 是否为升序？ {result}")  # 输出: True
