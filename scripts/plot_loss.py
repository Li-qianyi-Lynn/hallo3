#!/usr/bin/env python3
"""
从多个 train_4k_*.log 里提取 loss，按全局步数排列，画出曲线。

用法（在集群上）：
    python scripts/plot_loss.py
    python scripts/plot_loss.py --log_dir /scratch/li_qiany_neu/hallo3_logs
    python scripts/plot_loss.py --out loss_curve.png
"""

import re
import sys
import argparse
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", default="/scratch/li_qiany_neu/hallo3_logs")
    p.add_argument("--out", default="loss_curve.png")
    p.add_argument("--smooth", type=int, default=50, help="滑动平均窗口大小")
    return p.parse_args()

def extract_loss(log_path: Path):
    """
    从单个 log 文件里提取 (local_iter, loss) 列表。
    日志格式：
      iteration       74/    3200 | ... | total loss 5.846894E-02 | ...
    返回该文件内的局部步数和 loss。
    """
    records = []
    pattern = re.compile(
        r"iteration\s+(\d+)/\s*\d+.*?total loss\s+([\d.eE+\-]+)"
    )
    for line in log_path.read_text(errors="replace").splitlines():
        m = pattern.search(line)
        if m:
            records.append((int(m.group(1)), float(m.group(2))))
    return records

def smooth(values, window):
    out = []
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out

def main():
    args = parse_args()
    log_dir = Path(args.log_dir)

    logs = sorted(log_dir.glob("train_4k*.log"))
    if not logs:
        print(f"没找到 log 文件：{log_dir}/train_4k*.log")
        sys.exit(1)

    print(f"找到 {len(logs)} 个 log 文件：")
    for f in logs:
        print(f"  {f.name}")

    # 每个 log 文件的 iter 都从 1 开始（局部步数），
    # 按文件名顺序（时间顺序）拼接，每个文件加 offset
    iters = []
    losses = []
    global_offset = 0

    for f in logs:
        records = extract_loss(f)
        if not records:
            continue
        local_iters = [r[0] for r in records]
        local_losses = [r[1] for r in records]
        n = len(records)
        max_local = max(local_iters)
        print(f"  {f.name}: {n} 步, local {local_iters[0]}–{max_local}, global offset {global_offset}")
        for li, lv in zip(local_iters, local_losses):
            iters.append(global_offset + li)
            losses.append(lv)
        global_offset += max_local

    if not iters:
        print("未能解析到任何 loss，请确认日志格式。")
        print("手动试试：grep -i 'total loss' <log文件> | head -5")
        sys.exit(1)

    print(f"\n解析到 {len(iters)} 个数据点，全局步数范围：{iters[0]} – {iters[-1]}")
    print(f"Loss 范围：{min(losses):.4e} – {max(losses):.4e}")

    # 打印简单文字趋势（均匀采样 20 个点）
    print("\n=== Loss 趋势 ===")
    step = max(1, len(iters) // 20)
    for i in range(0, len(iters), step):
        bar = "#" * min(60, int(losses[i] * 800))
        print(f"  global ~{iters[i]:6d} | loss {losses[i]:.4e} | {bar}")

    # 画图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        smoothed = smooth(losses, args.smooth)

        # 用 95th percentile 截断 y 轴，避免尖峰把曲线压扁
        import statistics
        sorted_losses = sorted(losses)
        p95 = sorted_losses[int(len(sorted_losses) * 0.95)]
        ylim_top = min(p95 * 1.5, max(losses))

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # 左图：全量（包含尖峰）
        axes[0].plot(iters, losses, alpha=0.2, color="steelblue", linewidth=0.6, label="raw")
        axes[0].plot(iters, smoothed, color="steelblue", linewidth=2,
                     label=f"smoothed (w={args.smooth})")
        axes[0].set_title("Full range")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 右图：截断 y 轴，看清主趋势
        axes[1].plot(iters, losses, alpha=0.2, color="steelblue", linewidth=0.6, label="raw")
        axes[1].plot(iters, smoothed, color="steelblue", linewidth=2,
                     label=f"smoothed (w={args.smooth})")
        axes[1].set_ylim(0, ylim_top)
        axes[1].set_title(f"Zoomed (y ≤ {ylim_top:.3f}, p95×1.5)")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle("Training Loss Curve", fontsize=13)
        plt.tight_layout()
        plt.savefig(args.out, dpi=150)
        print(f"\n图已保存：{args.out}")
        print(f"把它 scp 到本地看：scp <cluster>:{Path(args.out).absolute()} .")
    except ImportError:
        print("\nmatplotlib 未安装，只输出文字趋势。")
        print("安装：pip install matplotlib")

if __name__ == "__main__":
    main()
