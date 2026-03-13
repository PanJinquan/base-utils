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


def solve_mixture(M, initial_concs, target_concs):
    N = len(initial_concs)
    if len(target_concs) != N:
        raise ValueError("初始浓度和目标浓度列表长度必须一致")

    # 1. 可行性预检查
    ratios = [t / i if i != 0 else float('inf') for t, i in zip(target_concs, initial_concs)]
    sum_ratios = sum(ratios)

    print(f"{'=' * 40}")
    print(f"可行性诊断:")
    print(f"目标/初始 比例之和 = {sum_ratios:.4f}")

    if abs(sum_ratios - 1.0) < 1e-4:
        print(">> 状态：完美可解 (Perfectly Solvable)")
        print(">> 可以直接通过解析解获得精确结果。")
    elif sum_ratios > 1.0:
        print(f">> 状态：原料过浓 (Over-concentrated)")
        print(f">> 解释：所需原料总质量约为最终质量的 {sum_ratios:.2f} 倍。")
        print(f">> 建议：需要加入约 {M * (sum_ratios - 1):.2f} 单位的溶剂（如水）来稀释，或者降低目标浓度。")
    else:
        print(f">> 状态：原料不足 (Under-concentrated)")
        print(f">> 解释：即使全部用完原料，也无法达到目标浓度。")
        print(f">> 建议：需要加入约 {M * (1 - sum_ratios):.2f} 单位的纯溶质，或者提高初始浓度/降低目标。")
    print(f"{'=' * 40}\n")

    # 2. 定义优化问题
    # 即使无解，我们也尝试寻找“误差最小”的配方
    def objective(vars):
        x = vars
        total_mass = sum(x)
        if total_mass == 0: return 1e9

        # 计算实际浓度
        actual_concs = [(x[i] * initial_concs[i]) / total_mass for i in range(N)]

        # 加权误差平方和 (这里简单使用平方和，也可以根据重要性加权)
        err = sum((actual_concs[i] - target_concs[i]) ** 2 for i in range(N))
        return err

    def constraint(vars):
        return sum(vars) - M

    # 初始猜测：使用理论比例，即使它们加起来不等于M，优化器会调整
    # 为了避免除以零，做个小保护
    x0 = []
    for t, i in zip(target_concs, initial_concs):
        if i == 0:
            x0.append(0.0)
        else:
            x0.append(M * (t / i) / sum_ratios)  # 归一化初始猜测

    # 优化求解
    sol = minimize(
        objective,
        x0,
        method='SLSQP',
        constraints={'type': 'eq', 'fun': constraint},
        bounds=[(0, M) for _ in range(N)],
        options={'ftol': 1e-9, 'maxiter': 1000}
    )

    x_opt = sol.x
    total_actual = sum(x_opt)

    # 计算实际达成的浓度
    actual_results = []
    if total_actual > 0:
        for i in range(N):
            c_act = (x_opt[i] * initial_concs[i]) / total_actual
            actual_results.append(c_act)
    else:
        actual_results = [0] * N

    # 输出详细结果
    print("优化结果详情:")
    print(f"{'物品':<6} | {'初始%':<8} | {'目标%':<8} | {'实际%':<8} | {'用量(g)':<10} | {'误差':<8}")
    print("-" * 70)

    max_err = 0
    for i in range(N):
        err = abs(actual_results[i] - target_concs[i])
        max_err = max(max_err, err)
        print(
            f"{i + 1:<6} | {initial_concs[i]:<8.2f} | {target_concs[i]:<8.2f} | {actual_results[i]:<8.2f} | {x_opt[i]:<10.2f} | {err:<8.2f}")

    print("-" * 70)
    if max_err > 1.0:  # 如果最大误差超过1%，认为结果不可用
        print("\n⚠️ 警告：由于物理限制，无法同时满足所有浓度目标。")
        print("   当前结果是数学上的‘最小误差解’，但在工程中可能不可用。")
        print("   请根据上方的‘可行性诊断’调整目标浓度或引入溶剂/溶质。")
    else:
        print("\n✅ 成功：所有浓度目标均在可接受误差范围内。")


# --- 测试用例 1: 你之前的“无解”数据 ---
print(">>> 测试案例 1: 原始数据 (无解情况)")
M = 1000
init_1 = [30, 40, 30, 25, 35]
tgt_1 = [20, 50, 30, 20, 25]
solve_mixture(M, init_1, tgt_1)

print("\n\n")

# --- 测试用例 2: 构造一个“有解”的数据 ---
# 设计思路: 让 sum(target/initial) = 1
# A: 30->15 (0.5), B: 40->10 (0.25), C: 20->10 (0.5) -> Sum = 1.25 (不行)
# 重新设计:
# A: 40->20 (0.5)
# B: 40->10 (0.25)
# C: 40->10 (0.25)
# Sum = 1.0 (完美)
print(">>> 测试案例 2: 构造的完美数据 (有解情况)")
M = 1000
init_2 = [40, 40, 40]
tgt_2 = [20, 10, 10]
solve_mixture(M, init_2, tgt_2)