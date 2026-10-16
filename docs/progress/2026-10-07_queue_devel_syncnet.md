# 2026-10-07：训练续训到 7000 步、排队问题与 rtx-devel、SyncNet 环境跑通

承接 [2026-10-02_4k_train_leak_isolation.md](2026-10-02_4k_train_leak_isolation.md)。

## 一、今日结论

| 项 | 结果 |
|----|------|
| 累计训练步数 | 7000 / 13600，剩余 6600 步 |
| `rtx-batch` 排队 | 排了一天没排上；原因是整个分区 8 张卡 × 17 节点全部占满，且账号 fairshare 很低 |
| 解决办法 | 改用 `rtx-devel` 分区（限时 4h），很快就分到卡；每轮手动申请 2 小时的交互节点，进去直接运行训练脚本 |
| `save_interval` | 300 → 250，减少每轮结束时作废的步数 |
| SyncNet | 环境已在集群上跑通 `demo_syncnet.py`（官方 example.avi） |

---

## 二、训练进度

### 各轮任务结果

| 任务 | 时长 | 结果 |
|------|------|------|
| `1163314`（10-02 起，4h） | `TIMEOUT 04:00:23` | 从累计 3400 续训，跑到本轮 1572 步，最后存的是 1500，作废 72 步；step 约 8.78 s；skip 20 条（约 0.64%）；没有 `decord child` 或 `Traceback` 报错 |
| `1214080`（10-06 起，6h） | — | 本轮存到 2100（目录 `10-06-23-37`）；sacct 结果、step 耗时和泄漏检查本次未收集 |

### 累计步数

各目录 `latest` 相加：`100 + 600 + 600 + 2100 + 1500 + 2100 = 7000`，剩余 `13600 - 7000 = 6600` 步，按 8.8 s/步约 16 小时。

### checkpoint 清理后的状态（约 275G）

| checkpoint | 累计步数 | 用途 |
|------------|----------|------|
| `10-06-23-37/2100` | 7000 | 续训起点 |
| `10-02-18-08/1500` | 4900 | 每轮终点，之后选 checkpoint 的候选 |
| `10-02-00-44/2100` | 3400 | 同上 |
| `10-01-19-00/600` | 1300 | 早期里程碑 |

`latest` 文件都不能删，累计步数靠它们相加。

---

## 三、排队问题

### 现象

提交 `1222641`（`rtx-batch`，6h）后排了约一天，`squeue --start` 的预计开始时间一直是 `N/A`。

### 排查结果

| 检查 | 结果 |
|------|------|
| `sprio` | 昨天 PRIORITY 224（FAIRSHARE 225，AGE 0）；今天降到 39（FAIRSHARE 39，AGE 1） |
| 队列前面的任务 | 优先级 640–940，至少 19 个以上 |
| `sinfo -p rtx-batch -N` | 17 个节点的 `GRES_USED` 全是 8/8，所有 136 张卡被占满 |
| 内存 | 每节点总内存约 2.3TB，已分配最多不到 1TB，不是瓶颈 |
| CPU | 每个节点都有几十个空闲核，不是瓶颈 |
| `sinfo -s` | `rtx-batch`、`b200-batch`、两个 devel、`preemptable` 的节点全部是已分配状态 |

结论：瓶颈是 GPU 总量，加上账号 fairshare 很低（最近连续跑了多个 2 卡任务），新提交的、fairshare 更高的任务会排在前面。`AGE`（排队时长）权重很小，干等几乎换不来优先级。

### 解决办法

改用 `rtx-devel` 分区（`a0018`、`a0019`，限时 4h），**每一轮手动申请一个 2 小时的交互节点，进去之后直接运行训练脚本**，不再走 `sbatch` 排队。`rtx-devel` 很快就能分到卡。

每一轮的操作：

1. 申请 2 卡、2 小时的交互节点：
   ```bash
   srun --partition=rtx-devel --nodes=1 --gpus=2 --cpus-per-task=16 --mem=256G --time=02:00:00 --account=p2026_0014_neu --pty bash
   ```
2. 分到节点后，在节点上运行（`NO_RESUBMIT=1` 必须加，同时把输出写到日志）：
   ```bash
   cd ~/hallo3
   NO_RESUBMIT=1 bash slurm/run_train_4k.sh 2>&1 | tee /scratch/li_qiany_neu/hallo3_logs/train_4k_manual_$(date +%m%d_%H%M).log
   ```
3. 约 2 小时后节点到期，训练被终止；然后清理这一轮多余的 checkpoint，再申请下一轮。

脚本里的续训逻辑照常工作：从各目录 `latest` 累计步数，从最新目录加载，`train_iters` 为剩余步数。每轮约 780 步，剩余 6600 步约需 8–9 轮。

### 注意

- **终端断开会杀掉训练**：交互 shell 里的进程随终端结束，所以要在 `tmux` / `screen` 里运行（登录节点上是否装了没有核实）。
- **不能没有 `NO_RESUBMIT=1`**：脚本里自动续排的 `sbatch` 不带 `--partition` 和 `--time`，会回到 `rtx-batch` 的 24h。
- **没有 Slurm 的日志文件**：脚本的 `#SBATCH --output` 在交互模式下不生效，所以用 `tee` 自己写日志，文件名里带时间，注意不要和之前的 `train_4k_<JOBID>.log` 混了。
- **旧任务要取消**：`rtx-batch` 里还在排的 `1222641` 如果没取消，哪天排上就会从同一个 checkpoint 再训一遍。
- **`USR1` 的提前通知不会触发**：2 小时到点节点直接收回，没有提前 3 分钟的信号，但因为已经加了 `NO_RESUBMIT=1`，没有影响；到点前最后一个 checkpoint 之后的步数作废。
- `rtx-devel` 的权限和每人并发上限没有专门查过，目前能用。

---

## 四、配置改动

`configs/sft_4k_train.yaml`：`save_interval` 300 → 250。

原因：2 小时约 780 步，按 300 只存 300、600，最后约 180 步作废（约 23% 算力）；按 250 存 250、500、750，最多作废三四十步。代价是每轮多存一个 69GB checkpoint。累计步数是各目录 `latest` 相加，不受影响。

---

## 五、SyncNet 环境（`syncnet` conda 环境）

在 `rtx-devel` 节点上申请 1 卡、小内存跑 demo：

```bash
srun --partition=rtx-devel --nodes=1 --gpus=1 --cpus-per-task=4 --mem=16G --time=00:30:00 --account=p2026_0014_neu --pty bash
python demo_syncnet.py --videofile data/example.avi --tmp_dir $HOME/tmp_syncnet
```

### 遇到的两个问题

| 问题 | 报错 | 原因 | 处理 |
|------|------|------|------|
| GPU 不兼容 | `CUDA error: no kernel image is available for execution on the device` | RTX PRO 6000 Blackwell（sm_120），环境里的 PyTorch 太旧，不含这个架构的 kernel | 升级到 CUDA 12.8 以上的 PyTorch 构建（`pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`），以官网安装选择器为准 |
| scipy 导入失败 | `libstdc++.so.6: version GLIBCXX_3.4.30 not found` | 运行时加载到了系统旧版 `libstdc++`，而不是 conda 环境里的 | `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH`；如环境里的库也太旧，则 `conda install -c conda-forge libstdcxx-ng` |

处理后 demo 在官方 `example.avi` 上跑通。**待补**：实际使用的是哪一种修复、输出的 AV offset / Min dist / Confidence 具体数值（README 参考值约为 3 / 5.353 / 10.021）。

### 评估自己的视频用完整流程

README 里另有完整流程，会先做人脸检测和裁切：

```bash
python run_pipeline.py --videofile /path/to/video.mp4 --reference name_of_video --data_dir /path/to/output
python run_syncnet.py  --videofile /path/to/video.mp4 --reference name_of_video --data_dir /path/to/output
```

`demo_syncnet.py` 只是演示脚本。是否会自动把帧率转成 25 fps（Hallo3 配置里视频是 8 fps）还没有核实，评估前要确认，否则音画偏移量会不准。

---

## 六、尚未完成 / 待确认

- **子进程隔离（方案 B）是否真的在生效**：`1163314` 和 `1214080` 都没有做 `top -H` 泄漏检查，也没有确认当时集群上跑的是不是 B 的代码（`grep -c _IsolatedReader` 和 `git log` 的提交时间）。间接证据：step 时间稳定、skip 比例相近、没有新报错
- `1222641` 在 `rtx-batch` 的旧任务是否已取消
- 今日 2 小时轮次的 jobid、日志开头（`累计 N/13600`）和结果
- SyncNet 在 Hallo3 生成视频上的评估：先在 val 集小批量试跑，确认帧率处理和人脸检测在生成视频上是否正常，再用 eval 集 350 条
- 对照组（原 wav2vec 编码器在同一份数据上训练同样步数）仍未开始
