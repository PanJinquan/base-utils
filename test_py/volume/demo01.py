# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : 2026-03-03 15:36:37
# @Brief  :
# --------------------------------------------------------
"""
import numpy as np
from scipy.optimize import minimize

# 参数输入
M = 1000  # 最终混合物质量
initial_concentrations = [30, 40, 30, 25, 35]  # 初始浓度列表（N个物品）
target_concentrations = [2, 5, 3, 5, 5]  # 目标浓度列表（N个物品）
N = len(initial_concentrations)  # 物品数量

# 目标函数：最小化浓度误差
def objective(vars):
    x = vars
    total_mass = sum(x)
    # 计算实际浓度
    actual_concentrations = [(x[i] * initial_concentrations[i]) / total_mass for i in range(N)]
    # 误差平方和
    err = sum((actual_concentrations[i] - target_concentrations[i])**2 for i in range(N))
    return err

# 约束条件：总质量为M
def constraint(vars):
    return sum(vars) - M

# 初始猜测值
x0 = [M * target_concentrations[i] / initial_concentrations[i] for i in range(N)]

# 优化求解
sol = minimize(
    objective,
    x0,
    method='SLSQP',
    constraints={'type': 'eq', 'fun': constraint},
    bounds=[(0, M) for _ in range(N)]  # 用量非负
)

# 输出结果
x_opt = sol.x
print("最优解：")
for i in range(N):
    print(f"物品{i+1}: {x_opt[i]:.2f}")
print("\n实际浓度：")
for i in range(N):
    print(f"物品{i+1}: {(x_opt[i] * initial_concentrations[i] / M):.2f}%")
