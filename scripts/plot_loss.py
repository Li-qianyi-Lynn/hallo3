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
    """从单个 log 文件里提取 (iteration, loss) 列表"""
    records = []
    pattern = re.compile(r"iteration\s+(\d+)[^\|]*\|\s*[Ll]oss[^\d]*([\d.e+\-]+)")
    # fallback: total loss
    pattern2 = re.compile(r"total loss[:\s]+([\d.e+\-]+).*iteration\s+(\d+)")
    pattern3 = re.compile(r"(\d+)/\d+.*?(?:total\s+)?loss[:\s]+([\d.e+\-]+)")

    for line in log_path.read_text(errors="replace").splitlines():
        m = pattern.search(line)
        if m:
            records.append((int(m.group(1)), float(m.group(2))))
            continue
        m = pattern2.search(line)
        if m:
            records.append((int(m.group(2)), float(m.group(1))))
            continue
        m = pattern3.search(line)
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

    # 合并所有 (iter, loss)，按 iter 排序去重
    all_records = []
    for f in logs:
        all_records.extend(extract_loss(f))

    if not all_records:
        print("未能解析到任何 loss，请确认日志格式。")
        print("手动试试：grep -i 'loss' <log文件> | head -5")
        sys.exit(1)

    all_records.sort(key=lambda x: x[0])
    # 去重（同一步可能出现在多个 log 里）
    seen = {}
    for it, loss in all_records:
        seen[it] = loss
    iters = sorted(seen.keys())
    losses = [seen[i] for i in iters]

    print(f"\n解析到 {len(iters)} 个数据点，步数范围：{iters[0]} – {iters[-1]}")
    print(f"Loss 范围：{min(losses):.4e} – {max(losses):.4e}")

    # 打印简单文字趋势（每 500 步取一个）
    print("\n=== Loss 趋势（每 500 步）===")
    step_size = 500
    prev_bucket = -1
    for it, loss in zip(iters, losses):
        bucket = it // step_size
        if bucket != prev_bucket:
            bar = "#" * int(loss * 500)  # 简单可视化
            print(f"  step {it:6d} | loss {loss:.4e} | {bar}")
            prev_bucket = bucket

    # 画图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        smoothed = smooth(losses, args.smooth)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(iters, losses, alpha=0.25, color="steelblue", linewidth=0.8, label="raw")
        ax.plot(iters, smoothed, color="steelblue", linewidth=2,
                label=f"smoothed (window={args.smooth})")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.out, dpi=150)
        print(f"\n图已保存：{args.out}")
        print(f"把它 scp 到本地看：scp <cluster>:{Path(args.out).absolute()} .")
    except ImportError:
        print("\nmatplotlib 未安装，只输出文字趋势。")
        print("安装：pip install matplotlib")

if __name__ == "__main__":
    main()
