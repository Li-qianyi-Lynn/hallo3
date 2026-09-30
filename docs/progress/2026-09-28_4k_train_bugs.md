# 4K 双卡训练：NCCL 超时排查与修复

更新日期：2026-09-28

背景：8 卡难排，改为 **2×B200 / 24h 分段续训**，目标约 10 epoch（`train_iters=13600`）。

---

## 一、训练计划调整

| 项目 | 原计划 (8 GPU) | 现计划 (2 GPU) |
|------|----------------|----------------|
| 分区 | `b200-batch` 8 卡 | `b200-batch` **2×B200** |
| 时限 | 12h | **24h** |
| steps | 5000 | **13600**（2719/2 × 10 epochs） |
| 预估总时长 | 4–7h | ~34h @ ~9s/step（约 2 个 24h session） |
| 续训 | 手动 | 超时前 USR1 自动 `sbatch`，启动时读最新 `latest` |

相关文件：
- `configs/sft_4k_train.yaml`
- `slurm/run_train_4k.sh`

---

## 二、失败记录

| JobID | 时长 | Exit | 现象 |
|-------|------|------|------|
| `1103249` | 31 min | 1 | NCCL ALLREDUCE timeout（SeqNum≈615） |
| `1104079` | 63 min | 1 | 同样 timeout（SeqNum≈1542），约卡在 **500 step** |
| smoke（交互 2×B200） | ~几分钟 | 0 | `iteration 2/2 \| loss ~5.27e-02`，流程可通 |

共同特征：
- Rank0 已 enqueue 下一次 `ALLREDUCE`，Rank1 仍停在上一次 → **双卡不同步**
- `Timeout(ms)=600000`（10 分钟）后 watchdog `SIGABRT`
- `train_output_4k/*/latest` 仍为 0（未到第一次有效存盘）

`sacct` 示例：`MaxRSS` 约 150–160GB，不是主机内存 OOM。

---

## 三、根因与修复

### Bug A：热路径 `[AUDIO_DEBUG]` print（第一次挂的主因之一）

每个 sample / forward 狂打 `print`，DataLoader worker + 两张卡抢 GIL，NCCL 提示可能是 **GIL deadlock**。交互只跑 2 步测不出来。

**修复：** `data_video.py` / `diffusion_video.py` / `transformer.py` 中 debug 默认关闭，需 `HALLO3_AUDIO_DEBUG=1` 才打印。

### Bug B：`forward_step_eval` 只让 DP rank0 跑 `log_video`（第二次挂的主因）

上游 Hallo3 初版（`6eb740a`）即如此，**不是后期测试误改**：

```python
# 旧逻辑（多卡 data parallel 会挂）
if mpu.get_data_parallel_rank() == 0:
    log_video(...)   # 含慢采样
model.shared_step(...)  # 含 allreduce
```

Rank0 还在采样，Rank1 已进 `allreduce` → 空等 10 分钟超时。  
`eval_interval=500` 时正好撞上第二次失败时间点。

**修复：** 两张卡都执行 `log_video`（写文件仍只在 `distributed.get_rank()==0`）。

### 配置侧规避（本轮训练）

| 项 | 旧值 | 新值 | 原因 |
|----|------|------|------|
| `eval_interval` | 500 | **99999** | 本轮先稳住训完，eval 非必须 |
| `save_interval` | 500 | **100** | 尽早有可续训 checkpoint |
| `num_workers` | 8 | **2** | 降低 Decord/worker 卡死风险 |
| 坏样本 | 直接炸 | `__getitem__` 跳过重试 | 单条坏视频拖死一张卡 |

说明：存盘靠 `save_interval`，与 eval **无关**；关掉 eval 不会导致不存盘。

---

## 四、当前状态

- 任务 `1107946`：`b200-batch` / `train-4k-2gpu` / **PD (Priority)**（以集群实时 `squeue` 为准）
- 数据：`/scratch/li_qiany_neu/talkvid_4k/` + `data/talkvid_4k.json`（2719 条）
- 输出：`/scratch/li_qiany_neu/train_output_4k/`
- 日志：`/scratch/li_qiany_neu/hallo3_logs/train_4k_<jobid>.log`

监控：

```bash
squeue -u li_qiany_neu
grep "total loss" /scratch/li_qiany_neu/hallo3_logs/train_4k_*.log | tail -5
cat /scratch/li_qiany_neu/train_output_4k/*/latest 2>/dev/null
```

排队位置（`b200-batch`）：

```bash
squeue -p b200-batch -t PD -o "%.10i %.8u %.6g %.10M %.20R" | grep -n "li_qiany"
```

---

## 五、变更文件

| 文件 | 改动 |
|------|------|
| `hallo3/train_video.py` | eval 时双卡都跑 `log_video` |
| `hallo3/data_video.py` | AUDIO_DEBUG 开关；坏样本跳过 |
| `hallo3/diffusion_video.py` | AUDIO_DEBUG 开关 |
| `hallo3/sgm/models/transformer.py` | AUDIO_DEBUG 开关 |
| `configs/sft_4k_train.yaml` | 13600 iters / eval 关闭 / save 100 / workers 2 |
| `slurm/run_train_4k.sh` | 2×B200、24h、续训、NCCL env |

---

## 六、下一步

1. 等 `1107946` 进入 `R`，确认能过 100/200 step 并写出 `latest`
2. 24h 到点后确认自动续训是否生效
3. 训完后再开 `eval_interval`（如 1000）做中间采样；代码侧双卡不同步已修
4. 继续 8K quality 数据准备（见 `splits/talkvid_en_8k_quality.json`）
