# 4K 双卡训练：decord 卡死导致 NCCL 超时（第三轮排查）

更新日期：2026-10-01

接续 `2026-09-28_4k_train_bugs.md`。上一轮修了 AUDIO_DEBUG 和 eval 双卡不同步，但训练仍然随机挂掉。本轮用 `py-spy` 抓到真正原因：**某条视频的 decord 取帧无限空转**。

---

## 一、本轮变化概览

| 项目 | 之前 | 现在 |
|------|------|------|
| 分区 | `b200-batch` | **`rtx-batch`**（RTX PRO 6000，`--gpus=2`），B200 devel 排不到 |
| 验证方式 | B200 交互节点 | `rtx-devel` 2 小时交互节点（job `1142191`，节点 `a0019`） |
| 续训进度统计 | 取步数最大的目录 | **各轮 `latest` 求和**（见第四节） |
| 视频读取 | 无保护 | 单样本 30s 超时 + 参考图避开视频尾部 |
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

共同现象：Rank 0 已进入下一次 allreduce，Rank 1 没跟上，10 分钟（`Timeout(ms)=600000`）后 watchdog `SIGABRT`。挂的步数每次不同，是因为取决于是否随机抽到问题视频的尾部帧。

---

## 三、根因：decord `seek_accurate` 在视频尾部帧无限空转

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

- 卡点是 `ref_image = vr[ref_idx]`（随机取参考图），`ref_idx=387`，离视频末尾只差 12 帧
- 取帧窗口 `vr.get_batch(...)` 正常，只有按下标取单帧（`seek_accurate`）卡住

### 验证结果（顺序解码）

```text
len(vr) = 399
You have received more than 100 frames corrupted and recovered from nearest frames.
sequentially decodable = 399
```

- 顺序解码能读满 399 帧，所以**不是"元数据帧数虚高"**（最初的推断被否定）
- decord 警告该视频有超过 100 帧损坏，是用相邻帧补上的，视频本身已损坏
- 随机 `seek_accurate` 落到损坏区域时会空转。卡死位置不一定只在最后 30 帧，所以下面的尾部保护**不保证够用**，真正兜底的是单样本超时，根治办法是把损坏视频从训练集里剔除（见第六节的扫描脚本）

### 因果链

```
抽到问题视频的尾部帧 -> rank1 worker 在 seek_accurate 空转
 -> rank1 拿不到 batch，GPU1 空闲
 -> rank0 在 allreduce 等 rank1
 -> 10 分钟后 NCCL watchdog 杀进程
```

`num_workers: 0` 不能解决这个问题，只是把卡死从 worker 挪到主进程，还会拖慢训练。

---

## 四、修复

### 0. 新增 `scripts/scan_bad_videos.py`

对训练 json 里每个视频在独立子进程中做探测（硬超时，因为 decord 空转无法从 Python 内打断）：顺序解码全部帧、检测 `corrupted` 警告、对头/中/尾若干固定下标做 `vr[i]` seek。输出 `<stem>_bad_videos.json`（路径 → 原因：`HANG` / `CORRUPT_FRAMES` / `LENMISMATCH` / `ERROR`）和剔除坏视频后的 `<stem>_clean.json`。

```bash
python scripts/scan_bad_videos.py --json data/talkvid_4k.json --workers 6
```

### 1. `hallo3/data_video.py`

- **`_TailSafeReader`**：包装 `VideoReader`，按下标取单帧时，下标落在最后 `REF_TAIL_MARGIN`（默认 30）帧内就在 `[0, len-31]` 里重新随机。`len()` 不变，音频下标缩放不受影响。环境变量：`HALLO3_REF_TAIL_MARGIN`
- **`_call_with_timeout`**：`__getitem__` 里单条样本读取放进线程，超过 `DECORD_TIMEOUT_S`（默认 30s，环境变量 `HALLO3_DECORD_TIMEOUT`）就放弃并换一条样本。decord 在原生代码里空转，`signal.alarm` 打断不了，所以用线程计时
- 超时或异常时打印 `[Stage2_SFTDataset] skip index=... path=...`，用于定位其他坏视频
- 已知代价：超时被放弃的线程会继续空转占一个核直到 worker 退出，所以不能只靠超时，要配合尾部保护

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

1. （已完成，结果见第三节）用 `VideoReader` 顺序 `next()` 数实际可解码帧数，对比 `len(vr)`：

```bash
python - <<'EOF'
from decord import VideoReader
p = "/scratch/li_qiany_neu/talkvid_4k/videos/videovideoMXAbum3V5bk-scene5-scene19.mp4"
print("len(vr) =", len(VideoReader(p, num_threads=1)))
vr = VideoReader(p, num_threads=1)
n = 0
try:
    while True:
        vr.next(); n += 1
except Exception as e:
    print("stopped by", type(e).__name__)
print("sequentially decodable =", n)
EOF
```

2. 本地改动未提交，集群上仍是旧版（HEAD `241e81f`）。需要 commit / push，再在集群 `git pull`，并确认：

```bash
grep -n "num_workers\|save_interval" configs/sft_4k_train.yaml
grep -n "_TailSafeReader" hallo3/data_video.py
```

3. 重新申请 `rtx-devel`，跑过 300 步并观察是否出现 `skip index=... path=...`
4. 通过后提交 `sbatch slurm/run_train_4k.sh`（先退出交互节点，避免与正式任务同时写 `train_output_4k`）
5. 若发现不止一条坏视频，写脚本一次性扫描 2719 条，提前过滤问题视频
6. 稳定后清理 `[DATA]` 心跳；训完再把 `eval_interval` 调回 1000 左右

---

## 七、变更文件

| 文件 | 改动 |
|------|------|
| `hallo3/data_video.py` | `_TailSafeReader`、`_call_with_timeout`、单样本 30s 超时、`num_threads=1`、跳过日志带视频路径 |
| `hallo3/train_video.py` | `[DATA]` 心跳（临时） |
| `configs/sft_4k_train.yaml` | `num_workers: 2`、`save_interval: 300` |
| `slurm/run_train_4k.sh` | `rtx-batch`、各轮 `latest` 求和续训、NCCL 与线程环境变量 |
| `docs/progress/2026-10-01_4k_train_decord_hang.md` | 本文档 |
