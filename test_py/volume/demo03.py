# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : 2026-03-03 15:56:22
# @Brief  :
# --------------------------------------------------------
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def solve_mixture_with_cost(M, initial_concs, target_concs, costs, weight=0.5):
    """
    求解混合问题，同时考虑浓度目标和成本

    参数:
    M: 最终混合物总质量 (克)
    initial_concs: 初始浓度列表 (长度N)
    target_concs: 目标浓度列表 (长度N)
    costs: 每个物品的成本列表 (长度N)
    weight: 浓度误差的权重 (0.0-1.0), 0=只考虑成本, 1=只考虑浓度

    返回:
    x_opt: 最优用量 (列表)
    actual_concs: 实际浓度 (列表)
    total_cost: 总成本
    concentration_error: 浓度误差平方和
    """
    N = len(initial_concs)
    if len(target_concs) != N or len(costs) != N:
        raise ValueError("初始浓度、目标浓度和成本列表长度必须一致")

    # 可行性诊断
    ratios = [t / i if i != 0 else float('inf') for t, i in zip(target_concs, initial_concs)]
    sum_ratios = sum(ratios)
    print(f"{'=' * 50}")
    print(f"可行性诊断: 目标/初始比例之和 = {sum_ratios:.4f}")
    if abs(sum_ratios - 1.0) < 1e-4:
        print(">> 状态：完美可解 (Perfectly Solvable)")
    elif sum_ratios > 1.0:
        print(f">> 状态：原料过浓 (Over-concentrated)")
        print(f">> 建议：需要加入约 {M * (sum_ratios - 1):.2f} 单位的溶剂")
    else:
        print(f">> 状态：原料不足 (Under-concentrated)")
        print(f">> 建议：需要加入约 {M * (1 - sum_ratios):.2f} 单位的纯溶质")
    print(f"{'=' * 50}\n")

    # 目标函数：加权浓度误差 + 成本
    def objective(vars):
        x = vars
        total_mass = sum(x)
        if total_mass < 1e-5:  # 避免除以零
            return 1e9

        # 计算实际浓度
        actual_concs = [(x[i] * initial_concs[i]) / total_mass for i in range(N)]

        # 浓度误差 (平方和)
        concentration_error = sum((actual_concs[i] - target_concs[i]) ** 2 for i in range(N))

        # 总成本
        total_cost = sum(x[i] * costs[i] for i in range(N))

        # 加权目标
        return weight * concentration_error + (1 - weight) * total_cost

    # 约束条件：总质量为M
    def constraint(vars):
        return sum(vars) - M

    # 初始猜测：使用理论比例（归一化）
    x0 = []
    for t, i in zip(target_concs, initial_concs):
        if i == 0:
            x0.append(0.0)
        else:
            x0.append(M * (t / i))
    # 归一化初始猜测
    if sum(x0) > 0:
        x0 = [xi / sum(x0) * M for xi in x0]
    else:
        x0 = [M / N] * N

    # 优化求解
    sol = minimize(
        objective,
        x0,
        method='SLSQP',
        constraints={'type': 'eq', 'fun': constraint},
        bounds=[(0, M) for _ in range(N)],
        options={'ftol': 1e-9, 'maxiter': 1000}
    )

    # 计算结果
    x_opt = sol.x
    total_actual = sum(x_opt)
    actual_concs = [(x_opt[i] * initial_concs[i]) / total_actual for i in range(N)]
    total_cost = sum(x_opt[i] * costs[i] for i in range(N))
    concentration_error = sum((actual_concs[i] - target_concs[i]) ** 2 for i in range(N))

    # 输出结果
    print("优化结果详情:")
    print(f"{'物品':<6} | {'初始%':<8} | {'目标%':<8} | {'实际%':<8} | {'用量(g)':<10} | {'成本(元)':<10}")
    print("-" * 70)
    for i in range(N):
        print(
            f"{i + 1:<6} | {initial_concs[i]:<8.2f} | {target_concs[i]:<8.2f} | {actual_concs[i]:<8.2f} | {x_opt[i]:<10.2f} | {x_opt[i] * costs[i]:<10.2f}")
    print("-" * 70)
    print(f"总成本: {total_cost:.2f} 元")
    print(f"浓度误差平方和: {concentration_error:.4f}")
    print(f"权重参数: λ = {weight:.2f}")

    # 生成浓度对比图
    plt.figure(figsize=(10, 6))
    x = np.arange(N)
    width = 0.35
    plt.bar(x - width / 2, target_concs, width, label='目标浓度')
    plt.bar(x + width / 2, actual_concs, width, label='实际浓度')
    plt.xlabel('物品')
    plt.ylabel('浓度 (%)')
    plt.title(f'浓度目标 vs 实际 (权重 λ={weight:.2f})')
    plt.xticks(x, [f'物品{i + 1}' for i in range(N)])
    plt.legend()
    plt.tight_layout()
    plt.show()

    return x_opt, actual_concs, total_cost, concentration_error


# =====================
# 测试用例
# =====================

if __name__ == "__main__":
    # 测试用例 1: 无解数据 + 成本优化
    print(">>> 测试案例 1: 无解数据 (比例和=4.43) + 成本优化")
    M = 1000
    initial_concs = [30, 40, 30, 25, 35]  # 初始浓度
    target_concs = [20, 50, 30, 20, 25]  # 目标浓度
    costs = [1.0, 0.8, 1.2, 0.9, 1.1]  # 成本 (元/克)

    # 测试不同权重
    weights = [0.0, 0.3, 0.5, 0.7, 1.0]
    results = []

    for w in weights:
        print(f"\n--- 权重 λ = {w:.2f} ---")
        _, _, total_cost, err = solve_mixture_with_cost(M, initial_concs, target_concs, costs, weight=w)
        results.append((w, total_cost, err))

    # 显示成本与权重的关系
    plt.figure(figsize=(10, 6))
    weights, costs, errors = zip(*results)
    plt.plot(weights, costs, 'bo-', label='总成本')
    plt.plot(weights, errors, 'ro-', label='浓度误差平方和')
    plt.xlabel('权重 λ (浓度重要性)')
    plt.ylabel('值')
    plt.title('成本与浓度误差随权重变化')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 测试用例 2: 有解数据 + 成本优化
    print("\n\n>>> 测试案例 2: 有解数据 (比例和=1.0) + 成本优化")
    M = 1000
    initial_concs = [40, 40, 40]  # 初始浓度
    target_concs = [20, 10, 10]  # 目标浓度
    costs = [1.0, 0.8, 1.2]  # 成本 (元/克)

    solve_mixture_with_cost(M, initial_concs, target_concs, costs, weight=0.5)