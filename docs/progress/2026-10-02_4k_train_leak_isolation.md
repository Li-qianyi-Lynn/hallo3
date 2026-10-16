# 2026-10-02：6 小时任务结果、decord 线程泄漏与子进程隔离

承接 [2026-10-01_4k_train_decord_hang.md](2026-10-01_4k_train_decord_hang.md)。

## 一、今日结论

| 项 | 结果 |
|----|------|
| 6h 任务（`1153680`，`rtx-batch`，2 卡） | 按时间限制结束（`TIMEOUT 06:00:28`），期间没有再出现 NCCL 超时，说明"线程超时 + 新 reader 读参考帧"已经挡住了卡死 |
| 累计训练步数 | 3400 / 13600（`100 + 600 + 600 + 2100`），剩余 10200 步 |
| 本轮损失的步数 | 本轮跑到第 2353 步，最后一个 checkpoint 是 2100（`save_interval: 300`），中间 253 步（约 37 分钟）因没有 checkpoint 而作废；每轮结束时最多损失 299 步 |
| 新发现 | 超时的 decord 调用不会真正结束，线程会一直空转占满一个核（CPU 泄漏） |
| 对策 | 方案 B：decord 读取放进 `os.fork` 出的子进程，超时直接 `SIGKILL`。代码已写好，**尚未提交、尚未在集群上验证** |

### 6h 任务（`1153680`）末段日志

| 项 | 数值 |
|----|------|
| sacct | `TIMEOUT`，Elapsed `06:00:28`，ExitCode `0:0`（`.batch` 为 `CANCELLED 0:15`，即被 Slurm 的时限信号终止） |
| 前一个任务 `1153563` | `CANCELLED`，Elapsed `00:00:00`，从未运行 |
| 结束时进度 | iteration 2353 / 12300（本轮计数，目标是 `train_iters` 的剩余值） |
| step 耗时 | 约 8.85 s/iter（8.84–9.05 s），6 小时后仍稳定，没有出现因泄漏线程变慢的迹象 |
| 吞吐 | 约 6.76 samples/(min·GPU) |
| 学习率 | 8.19e-6（本轮 cosine 衰减中的位置；每轮续训会重新开始 warmup 和衰减） |
| loss | 单步 0.04–0.2，波动大，需看均值，不能凭单步判断 |

6 小时内泄漏线程没有把 17 核占满，所以 step 时间没变慢；但按 1.5 个 / 小时估算，24 小时单进程会累积到约 36 个，会严重影响速度，因此仍需要方案 B。

## 二、线程泄漏

### 现象

训练运行约 4 小时后，进入计算节点（`a0011`，17 核）执行 `top -H -bn1 -u $USER`：

- 2 个训练 autograd 线程
- 6 个 `pt_data_worker` 线程，各约 94% CPU，累计 CPU 时间 242 / 242 / 121 / 77 / 36 / 21 分钟
- 节点 load 约 27.8；此时 step 时间仍是 8.8 s/iter，尚未明显变慢

### 原因

第一版超时用线程实现（`_call_with_timeout`）：超时后只是丢弃结果，线程本身仍在 decord 的原生代码里。Python 无法终止原生调用，其中约 35% 永远不返回，持续空转。

### 估算

- 约 1.5 个线程 / 小时被永久占住
- 17 核大约 6–10 小时后被占满，24h 任务后半段会越来越慢
- 泄漏只在进程重启时清零

### 为什么不是数据本身的问题

- 取一个坏视频肉眼检查，画面正常
- 坏 clip 分布：32 条坏 clip 来自 26 个来源（共 1229 个来源，受影响来源里共有 102 条 clip），基本是单条独立的，不是按来源整体损坏
- 目前已剔除 17 条（`talkvid_4k_clean.json` 2702 / 2719），skip 率仍约 0.5–0.67%，估计总共约 40 条
- 视频损坏的具体原因（时间戳 / GOP 结构）还没查清，重新编码是否有效尚未测试

## 三、方案 B：子进程隔离

改动在 `hallo3/data_video.py`：

- `_decord_isolated(video_path, op, arg)`：`os.fork()` 子进程，用全新 `VideoReader` 执行 `len` / `window` / `frame`，结果以 numpy 经管道传回；超过 `DECORD_TIMEOUT_S`（默认 30s，环境变量 `HALLO3_DECORD_TIMEOUT`）则 `SIGKILL` 并 `waitpid`
- `_IsolatedReader`：对外接口与 `VideoReader` 一致（`len(vr)`、`vr.get_batch(...)`、`vr[idx]`），替换 `Stage2_SFTDataset` 两个分支（bbox / mask）里的 `VideoReader`
- 不用 `multiprocessing.Process`：DataLoader worker 是 daemon 进程，不允许再建子进程
- 删除线程版 `_call_with_timeout` 和 `_read_frame_fresh_reader`（每次读取本来就用全新 reader）
- skip 日志格式不变（`skip index=... path=...: TimeoutError`），`exclude_videos_from_log.py` 无需改动
- 起点、参考帧下标仍在父进程里随机抽，不存在子进程复制 RNG 状态导致重复采样的问题

本地用假 decord 的测试：正常返回、异常透传、挂死 2s 超时后被杀且无残余子进程、300MB 传输 0.21s、单次 fork 约 1ms、daemon 进程内可用。

**未验证项**：真实 decord 在子进程里的行为（`bridge.set_bridge("native")` + `.asnumpy()`）、集群上 fork 带来的内存与 CPU 开销、真实 step 时间。

## 四、checkpoint 状态

清理前（共 275G，4 个 checkpoint，每个约 69G）：

| 目录 | 内容 | 处理 |
|------|------|------|
| `10-01-03-30` | 仅 `latest=100` 和配置 | 保留（累计步数靠 `latest`） |
| `10-01-15-17` | 仅 `latest=600` 和配置 | 保留 |
| `10-01-19-00` | `600` | 保留，作为里程碑 |
| `10-02-00-44` | `1500`、`1800`、`2100`，`latest=2100` | 保留 `2100`，删除 `1500`、`1800` |

清理后应只剩 `19-00/600` 和 `00-44/2100`，约 138G。注意：各目录的 `latest` 文件不能删，续训脚本靠它们累加步数。

续训后日志应显示 `续训: load=.../train-4k-ltx-vae-10-02-00-44  累计 3400/13600`。

## 五、下一步

1. 提交并 push `data_video.py` 的改动，集群 pull
2. 提交 2 小时验证任务：`sbatch --time=02:00:00 --export=ALL,NO_RESUBMIT=1 slurm/run_train_4k.sh`
3. 跑 1 小时以上后，在运行节点检查：
   ```bash
   top -H -bn1 -u $USER | head -22
   ps -eo stat,ppid,pid,cmd | awk '$1 ~ /^Z/' | head
   grep -c "skip index" /scratch/li_qiany_neu/hallo3_logs/train_4k_<JOBID>.log
   ```
   通过标准：无 `pt_data` 线程长期 90% 以上 CPU，无 zombie，step 时间约 8–9 s 且稳定，skip 比例约 0.5%
4. 通过后提交 24 小时正式任务：`sbatch slurm/run_train_4k.sh`
5. 提交前用 `exclude_videos_from_log.py` 把 `train_4k_1153680.log` 中新增的坏视频并入 `talkvid_4k_clean.json`

## 六、待办 / 可选

- 重新编码坏视频（ffmpeg）是否能消除卡死，还没测；若有效可恢复被剔除的 clip，并避免数据损失
- 去掉 `train_video.py` 里临时的 `[DATA]` 心跳打印
- 没有真正的验证集（`valid_data = train_data`，`eval_interval: 99999`）；如需比较，可对里程碑 checkpoint（如 `19-00/600`）做固定样本生成对比
