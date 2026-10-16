# 4K 双卡训练：decord 卡死导致 NCCL 超时（第三轮排查）

更新日期：2026-10-01

接续 `2026-09-28_4k_train_bugs.md`。上一轮修了 AUDIO_DEBUG 和 eval 双卡不同步，但训练仍然随机挂掉。本轮用 `py-spy` 抓到真正原因：**含大量损坏帧的视频上，decord 单帧 seek 极慢甚至卡死**。

---

## 一、本轮变化概览

| 项目 | 之前 | 现在 |
|------|------|------|
| 分区 | `b200-batch` | **`rtx-batch`**（RTX PRO 6000，`--gpus=2`），B200 devel 排不到 |
| 验证方式 | B200 交互节点 | `rtx-devel` 2 小时交互节点（job `1142191`，节点 `a0019`） |
| 续训进度统计 | 取步数最大的目录 | **各轮 `latest` 求和**（见第四节） |
| 视频读取 | 无保护 | 单样本 30s 超时 + 扫描并剔除坏视频 |
| `num_workers` | 0（临时） | 2 |
| `save_interval` | 100 / 200 | **300** |

---

## 二、失败记录（累计）

| 任务 | 现象 | 结论 |
|------|------|------|
| `1103249`（31 min） | NCCL ALLREDUCE 超时 | 热路径 AUDIO_DEBUG print（已关） |
| `1104079`（63 min，约 500 步） | 同上 | eval 只让 DP rank0 跑 `log_video`（已修） |
| `1107946`（第 16 步后 10 min） | 同上，SeqNum 186 | 当时怀疑 DataLoader / decord，已加 `num_workers: 0`，**方向对但没找到点** |
| 交互 `1142191`（第 176 步） | 日志停住，将被 watchdog 杀 | **用 py-spy 定位到具体视频** |

共同现象：Rank 0 已进入下一次 allreduce，Rank 1 没跟上，10 分钟（`Timeout(ms)=600000`）后 watchdog `SIGABRT`。挂的步数每次不同，是因为取决于是否随机抽到问题视频、且随机帧落在难读的区域。

---

## 三、根因：损坏视频上 decord `seek_accurate` 极慢/卡死

### 现场（第 176 步，04:01:19 EDT 起停住）

`py-spy dump` 结果：

| 进程 | 状态 | 位置 |
|------|------|------|
| rank0 主进程 | active，GPU0 利用率 100% | `train_step`（等 NCCL allreduce） |
| rank1 主进程 | idle，GPU1 利用率 **0%** | `forward_step (train_video.py:267)` → `next(data_iterator)` → `queue.get` |
| rank1 DataLoader worker（PID 2239726） | active，CPU 约 43% | decord `seek_accurate` ← `vr[idx]` ← `_getitem_impl (data_video.py:894)` |

### 具体视频（`py-spy dump --pid <worker> --locals`）

```text
video_path: /scratch/li_qiany_neu/talkvid_4k/videos/videovideoMXAbum3V5bk-scene5-scene19.mp4
index: 1021     ori_vlen: 399
start: 223  end: 272        # 取 49 帧的窗口，解码成功
idx: 387                    # data_video.py:894 的 vr[ref_idx]，卡住
```

- 卡点是 `ref_image = vr[ref_idx]`（随机取参考图），`ref_idx=387`
- 取帧窗口 `vr.get_batch(...)` 正常，只有按下标取单帧（`seek_accurate`）卡住

### 验证结果（顺序解码）

```text
len(vr) = 399
You have received more than 100 frames corrupted and recovered from nearest frames.
sequentially decodable = 399
```

- 顺序解码能读满 399 帧，所以**不是"元数据帧数虚高"**（最初的推断被否定）
- decord 警告该视频有超过 100 帧损坏，是用相邻帧补上的，视频本身已损坏
- 视频本身已损坏，顺序解码没问题，但随机 seek 会出问题

### 验证结果（逐帧 seek 计时，每次全新进程）

```text
idx 0   0.39s     idx 240  15.77s
idx 60  0.55s     idx 300  0.09s
idx 120 0.70s     idx 340  0.15s
idx 180 8.81s     idx 368/380/387/398  0.20~0.24s
get_batch([387]) 0.24s,  get_batch([380]) 0.35s
```

- **尾部帧不特殊**：`vr[387]` 全新 reader 只要 0.23s，最初"尾部帧有问题"的推断被否定，已撤掉对应的尾部保护代码
- **中段 seek 极慢**：下标 180、240 分别用了 8.8s、15.8s，其余都是零点几秒，说明损坏段内没有可用关键帧，decord 要硬解很多帧
### 验证结果（按训练顺序复现，各 30s 超时）

训练里先 `get_batch(223..272)` 取窗口，再取参考帧 387：

| 写法 | 结果 |
|------|------|
| 同一 reader `vr[387]`（训练原写法） | **超时 >30s**，Ctrl-C 也打断不了 |
| 同一 reader `get_batch([387])` | **超时 >30s** |
| **新开 reader `VideoReader(p)[387]`** | **0.24s** |

- 结论：卡死与 **reader 状态**有关。`get_batch` 取完窗口后，同一个 reader 再往后 seek 会在损坏段里卡死；新 reader 从头 seek 则正常
- 卡死的原生调用不响应 SIGINT，Python 层的超时只能放弃样本，无法终止卡住的线程
- 新 reader 在 idx 180 / 240 仍分别要 8.8s / 15.8s（慢但会返回），所以仍需 30s 超时兜底和坏视频扫描

### 因果链

```
抽到损坏视频，随机帧落在难读区域 -> rank1 worker 在 seek_accurate 里极慢/卡死
 -> rank1 拿不到 batch，GPU1 空闲
 -> rank0 在 allreduce 等 rank1
 -> 10 分钟后 NCCL watchdog 杀进程
```

`num_workers: 0` 不能解决这个问题，只是把卡死从 worker 挪到主进程，还会拖慢训练。

---

## 四、修复

### 0. 新增 `scripts/scan_bad_videos.py`

对训练 json 里每个视频在独立子进程中做探测（硬超时，因为 decord 卡死无法从 Python 内打断）：顺序解码全部帧、检测 `corrupted` 警告，再按训练现在的读法逐次计时（3 个 49 帧窗口用 `get_batch`，12 个均匀分布的单帧各用新 reader 读），单次超过 `--slow`（默认 3s）判为慢。输出 `<stem>_bad_videos.json`（路径 → 原因：`HANG` / `SLOW_SEEK` / `CORRUPT_FRAMES` / `LENMISMATCH` / `ERROR`）和剔除坏视频后的 `<stem>_clean.json`。

```bash
python scripts/scan_bad_videos.py --json data/talkvid_4k.json --workers 6
```

### 1. `hallo3/data_video.py`

- **`_decord_isolated` + `_IsolatedReader`（子进程隔离，第二版）**：Stage2 数据集两个分支（bbox / mask）里的 `VideoReader` 换成 `_IsolatedReader`，用法不变（`len(vr)`、`vr.get_batch(...)`、`vr[ref_idx]`），但每次调用都 `os.fork()` 出子进程，用全新的 `VideoReader` 解码，结果以 numpy 经管道传回；超过 `DECORD_TIMEOUT_S`（默认 30s，环境变量 `HALLO3_DECORD_TIMEOUT`）就对子进程 `SIGKILL` 并 `waitpid` 回收，抛 `TimeoutError`，由 `__getitem__` 换一条样本
- 不用 `multiprocessing.Process`：DataLoader worker 是 daemon 进程，不允许再建子进程，所以直接 `os.fork`
- 每次读取用全新 reader，也满足"窗口读完后同一个 reader 再 seek 会卡死"的限制，参考帧不再需要单独处理
- 超时或异常时仍打印 `[Stage2_SFTDataset] skip index=... path=...: TimeoutError: ...`，`scripts/exclude_videos_from_log.py` 的正则不用改
- 数据内容不变：分辨率相同（`height=-1, width=-1`），dtype 仍是 uint8 的 `[T,H,W,3]`；随机数（起点、参考帧下标）仍在父进程里抽，不存在子进程 RNG 复制问题

#### 为什么要从线程超时换成子进程（第一版的缺陷）

第一版用线程超时（`_call_with_timeout`）只是"放弃"线程，而 Python 无法终止原生调用。在运行中的节点上用 `top -H -u $USER` 看到 6 个 `pt_data_worker` 线程各占约 94% CPU，累计 CPU 时间 242 / 242 / 121 / 77 / 36 / 21 分钟，约 35% 的超时调用永远不返回，空转泄漏约 1.5 个核 / 小时。节点只有 17 核，约 6–10 小时后会被占满，24h 任务后半段会越来越慢。泄漏只在进程重启时清零。子进程被 SIGKILL 后整个进程（含 decord 线程）一起消失，所以不再泄漏。

本地模拟测试（假 decord）：正常返回、异常透传、挂死 2s 超时后被杀且无残余子进程、300MB 数据传输 0.21s、单次 fork 开销约 1ms、daemon 进程内可用。

- 曾加过"参考图避开最后 30 帧"的 `_TailSafeReader`，因尾部帧并不特殊，已撤销

### 2. `configs/sft_4k_train.yaml`

| 项 | 值 | 说明 |
|----|----|------|
| `num_workers` | 2 | 恢复预取；之前设 0 是基于错误判断 |
| `save_interval` | 300 | 每次存盘约 75s，约 44 分钟一存 |
| `train_iters` | 13600 | 不变 |
| `eval_interval` | 99999 | 不变 |

### 3. `slurm/run_train_4k.sh`

- 分区 `rtx-batch`，`--gpus=2`，`--cpus-per-task=16`，`--mem=256G`，`--time=24:00:00`
- **续训步数修正**：SAT `finetune` 模式每次启动步数从 0 计，每轮会新建实验目录 `{save}/{experiment_name}-{MM-DD-HH-MM}`，其中 `latest` 只记录**该轮**最后存盘步数。原脚本取"步数最大的目录"会丢掉第二轮之后的进度。现改为 `scan_ckpts`：
  - 对所有 `latest` 求和得到 `TOTAL_DONE`
  - `REMAINING = TRAIN_ITERS - TOTAL_DONE`，`<=0` 则退出
  - 加载 mtime 最新的目录，overlay 里设 `train_iters: REMAINING`
  - 训练后再扫一次，未完成才 `safe_resubmit`；训练非 0 退出不自动重提
- 环境变量：`PYTHONUNBUFFERED=1`、`NCCL_DEBUG=WARN`、`TORCH_NCCL_ASYNC_ERROR_HANDLING=1`、`TORCH_NCCL_TRACE_BUFFER_SIZE=1000`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`

### 4. `hallo3/train_video.py`

- `forward_step` 里临时的 `[DATA] waiting for next batch / got batch` 心跳，调试用，稳定后可删

---

## 五、交互测试中确认的其他事实

### 学习率日志 `5.000E-07`（配置是 `1e-5`）不是 bug

SAT 的 `get_learning_rate_scheduler` 对非 pretrain 模式有 `auto_warmup_steps=100, auto_warmup_rate=0.05`：加载权重后前 100 步 lr = 0.05 × 1e-5 = 5e-7，之后再按 `warmup`（0.01 × train_iters）升到 1e-5，再线性衰减。第 101 步 lr=7.5e-6，符合预期。

### 第 101 步耗时 83.8s 不是卡死

那是第 100 步的存盘。单个 checkpoint **69GB**（`mp_rank_00_model_states.pt`），写一次约 75s。第 1 步 22s 是启动开销。

### checkpoint 空间

- SAT 的 `arguments.py` 里没有 `max_save` / `keep` 之类选项，**不会自动清理旧 checkpoint**
- `/scratch` 总量 2.7P，剩余 2.4P，整体空间不是问题；个人配额未知
- `save_interval: 300` 全程约 45 个 checkpoint，约 3.1TB。若有配额限制，再加定时清理旧目录的逻辑

---

## 六、待验证与下一步

1. （已完成，结果见第三节）顺序解码与逐帧 seek 计时均已验证
2. （已完成，结果见第三节）按训练顺序复现，确认新开 reader 读参考帧可绕开卡死

3. commit / push 本轮改动，集群 `git pull` 后运行 `scripts/scan_bad_videos.py`（集群上才有数据和 json），得到 `_bad_videos.json` 与 `_clean.json`
4. 把 `sft_4k_train.yaml` 的数据 json 换成 `_clean.json`
5. 重新申请 `rtx-devel`，跑过 300 步并观察是否出现 `skip index=... path=...`
6. 通过后提交 `sbatch slurm/run_train_4k.sh`（先退出交互节点，避免与正式任务同时写 `train_output_4k`）
7. 稳定后清理 `[DATA]` 心跳；训完再把 `eval_interval` 调回 1000 左右

---

## 七、变更文件

| 文件 | 改动 |
|------|------|
| `hallo3/data_video.py` | `_call_with_timeout`、单样本 30s 超时、`num_threads=1`、跳过日志带视频路径 |
| `scripts/scan_bad_videos.py` | 新增：扫描损坏 / 慢 seek / 卡死的视频并生成 `_clean.json` |
| `hallo3/train_video.py` | `[DATA]` 心跳（临时） |
| `configs/sft_4k_train.yaml` | `num_workers: 2`、`save_interval: 300` |
| `slurm/run_train_4k.sh` | `rtx-batch`、各轮 `latest` 求和续训、NCCL 与线程环境变量 |
| `docs/progress/2026-10-01_4k_train_decord_hang.md` | 本文档 |

---

## 八、第一次 sbatch（`1143441`）失败：三个脚本/加载问题

200 步交互测试通过后提交 24h 任务，`rtx-batch` 3 分 23 秒后 exit 1。

| 问题 | 现象 | 修复 |
|------|------|------|
| 两张卡分到两个节点 | `NodeList=a[0006,0010]`，日志 `GPUs: 1`；脚本只有 `--gpus=2`，没有 `--nodes=1` | 加 `--nodes=1 --ntasks=1`；启动后检查本节点卡数不是 2 就退出 |
| 续训加载 checkpoint 被拦 | `UnpicklingError: Weights only load failed`（torch>=2.6 默认 `weights_only=True`）；SAT 存的 checkpoint 含 `optimizer`、`lr_scheduler`、随机数状态等非张量内容。从预训练起步时文件只有权重，所以此前没暴露 | `train_video.py` 的 `_patched_load_checkpoint` 里临时把 `torch.load` 默认改成 `weights_only=False`；脚本加 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` |
| 自动重提交可能不触发 | bash 在前台命令（torchrun）运行期间不执行 trap，USR1 要等训练结束才处理，超时被杀时来不及重提交 | torchrun 放后台，用 `wait` 循环等待，USR1 可及时触发 |
| 重提交的新任务可能与当前任务并行 | `sbatch` 无依赖，新任务可能在别处抢到卡，从同一 checkpoint 同时训练 | 改为 `sbatch --dependency=afterany:$SLURM_JOB_ID` |

附加：脚本支持 `NO_RESUBMIT=1`，用于短时验证任务，到点前不自动排下一轮。

验证 checkpoint 可加载（登录节点，`mmap` 懒加载）：默认加载失败，`weights_only=False` 后可读出 `module`（2894 个张量）及 `optimizer` 等键。

续训路径确认可用：日志打印 `续训: load=.../train-4k-ltx-vae-10-01-03-30  累计 100/13600`。

2 小时验证任务：

```bash
sbatch --time=02:00:00 --export=ALL,NO_RESUBMIT=1 slurm/run_train_4k.sh
```

约 780 步，会存 300、600 两个 checkpoint；到点前 3 分钟日志应出现 `USR1` 和 `NO_RESUBMIT=1，跳过`，说明信号处理已生效。通过后正式提交：`sbatch slurm/run_train_4k.sh`。

---

## 九、6 小时任务（`1153680`）观察与 checkpoint 清理规则

- 约 1528 / 12300 步时 8.8 s/iter，约 0.5–0.67% 的样本触发 skip；已有 17 条坏视频被 `exclude_videos_from_log.py` 从 `talkvid_4k_clean.json` 剔除（2702 / 2719），估计总共约 40 条
- 坏视频基本是单条独立的（32 条坏 clip 分布在 26 个来源里），不是按来源整体损坏，肉眼看视频也正常
- 发现线程泄漏（见第四节 1），改为子进程隔离
- checkpoint 单个约 69GB，清理规则：每个目录只保留 `latest` 指向的那一步，另外保留 `10-01-19-00/600` 作为里程碑，其余删除

### 子进程隔离上线后的验证

训练起来后，在运行节点上检查：

```bash
top -H -bn1 -u $USER | head -22     # 不应再出现多个 pt_data 线程长期 90%+ CPU
grep -c "skip index" /scratch/li_qiany_neu/hallo3_logs/train_4k_<JOBID>.log
ps -eo stat,ppid,pid,cmd | awk '$1 ~ /^Z/' | head   # 不应有 zombie
```

通过标准：运行数小时后仍无累计空转线程，step 时间稳定，skip 数量与之前相近（约 0.5%）。
