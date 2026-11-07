#!/usr/bin/env python3
"""
分析保存的H_res矩阵

用法:
    python analyze_h_res_matrices.py <h_res_dir>

例如:
    python analyze_h_res_matrices.py ./outputs/checkpoints/h_res_matrices/
"""

import json
import sys
from pathlib import Path
import numpy as np


def load_h_res_step(step_dir):
    """加载某一步的H_res矩阵"""
    json_path = step_dir / "h_res_matrices.json"
    if not json_path.exists():
        return None

    with open(json_path, 'r') as f:
        data = json.load(f)

    # 转换为numpy数组
    h_res_matrices = {}
    for key, value in data.items():
        h_res_matrices[key] = np.array(value)

    return h_res_matrices


def analyze_h_res_matrix(h_res, name):
    """分析单个H_res矩阵的属性"""
    print(f"\n{name}:")
    print(f"  Shape: {h_res.shape}")
    print(f"  对角线值: {np.diag(h_res)}")
    print(f"  对角线均值: {np.mean(np.diag(h_res)):.6f}")

    # 计算非对角线元素
    mask = np.ones_like(h_res, dtype=bool)
    np.fill_diagonal(mask, False)
    offdiag = h_res[mask]

    print(f"  非对角线均值: {np.mean(offdiag):.6f}")
    print(f"  非对角线标准差: {np.std(offdiag):.6f}")
    print(f"  非对角线范围: [{np.min(offdiag):.6f}, {np.max(offdiag):.6f}]")

    # 矩阵范数
    print(f"  Frobenius范数: {np.linalg.norm(h_res, 'fro'):.6f}")
    print(f"  最大奇异值: {np.linalg.svd(h_res, compute_uv=False)[0]:.6f}")

    return {
        'diag_mean': np.mean(np.diag(h_res)),
        'offdiag_mean': np.mean(offdiag),
        'offdiag_std': np.std(offdiag),
        'frobenius_norm': np.linalg.norm(h_res, 'fro'),
    }


def compare_steps(h_res_dir):
    """比较不同步数的H_res矩阵变化"""
    h_res_dir = Path(h_res_dir)

    if not h_res_dir.exists():
        print(f"错误: 目录不存在: {h_res_dir}")
        return

    # 找到所有步数目录
    step_dirs = sorted([d for d in h_res_dir.iterdir() if d.is_dir() and d.name.startswith('h_res_step_')])

    if not step_dirs:
        print(f"错误: 在 {h_res_dir} 中没有找到H_res数据")
        return

    print(f"找到 {len(step_dirs)} 个保存的步数")
    print("=" * 60)

    # 分析每一步
    all_stats = {}
    for step_dir in step_dirs:
        step = int(step_dir.name.split('_')[-1])
        print(f"\n步数 {step}:")
        print("-" * 60)

        h_res_matrices = load_h_res_step(step_dir)
        if h_res_matrices is None:
            print(f"  警告: 无法加载数据")
            continue

        step_stats = {}
        for name, h_res in sorted(h_res_matrices.items()):
            stats = analyze_h_res_matrix(h_res, name)
            step_stats[name] = stats

        all_stats[step] = step_stats

    # 汇总分析
    if len(all_stats) > 1:
        print("\n" + "=" * 60)
        print("变化趋势分析")
        print("=" * 60)

        steps = sorted(all_stats.keys())
        first_step = steps[0]
        last_step = steps[-1]

        print(f"\n从步数 {first_step} 到 {last_step}:")

        # 选择第一个矩阵进行比较
        sample_name = list(all_stats[first_step].keys())[0]

        print(f"\n以 {sample_name} 为例:")
        first_stats = all_stats[first_step][sample_name]
        last_stats = all_stats[last_step][sample_name]

        print(f"  对角线均值变化: {first_stats['diag_mean']:.6f} -> {last_stats['diag_mean']:.6f}")
        print(f"  非对角线均值变化: {first_stats['offdiag_mean']:.6f} -> {last_stats['offdiag_mean']:.6f}")
        print(f"  非对角线标准差变化: {first_stats['offdiag_std']:.6f} -> {last_stats['offdiag_std']:.6f}")

        # 跨层比较（最后一步）
        print(f"\n最后一步 ({last_step}) 跨层比较:")
        for name, stats in sorted(all_stats[last_step].items()):
            print(f"  {name}:")
            print(f"    对角线均值: {stats['diag_mean']:.6f}")
            print(f"    非对角线均值: {stats['offdiag_mean']:.6f}")


def main():
    if len(sys.argv) != 2:
        print("用法: python analyze_h_res_matrices.py <h_res_dir>")
        print("\n例如:")
        print("  python analyze_h_res_matrices.py ./outputs/checkpoints/h_res_matrices/")
        sys.exit(1)

    h_res_dir = sys.argv[1]
    compare_steps(h_res_dir)


if __name__ == "__main__":
    main()
