# Slurm 集群作业管理

集群：NEU Explorer (`login.explorer.northeastern.edu`)  
用户：`li.qianyi`

---

## sbatch 基本用法

`sbatch` 把脚本提交给集群排队运行，**不占终端、关掉 ssh 也继续跑**。

```bash
sbatch your_script.sh
# → Submitted batch job 8499650
```

### sbatch 脚本头部参数说明

```bash
#!/bin/bash
#SBATCH -p short              # 分区（short / sharing / gpu 等）
#SBATCH --mem=32G             # 内存
#SBATCH --cpus-per-task=16    # CPU 核数
#SBATCH --gres=gpu:h200:4     # GPU（类型:数量）
#SBATCH -t 48:00:00           # 最长运行时间 HH:MM:SS
#SBATCH -o logs/job_%j.log    # 日志路径（%j = job ID）
```

# example
  cat > /scratch/li.qianyi/TalkVid/run_download.sh << 'EOF'
  #!/bin/bash
  #SBATCH -p short
  #SBATCH --mem=32G
  #SBATCH --cpus-per-task=16
  #SBATCH -t 48:00:00
  #SBATCH -o /scratch/li.qianyi/TalkVid/logs/download_%j.log

  cd /scratch/li.qianyi/TalkVid/data_pipeline/0_video_download

  python download_clips.py \
      --input /scratch/li.qianyi/TalkVid/data/filtered_video_clips_with_captions.json \
      --output /scratch/li.qianyi/TalkVid/clips_download \
      --workers 8 \
      --cookies /scratch/li.qianyi/TalkVid/youtube_cookies.txt
  EOF

  sbatch /scratch/li.qianyi/TalkVid/run_download.sh

### 常用查询命令

```bash
squeue -u li.qianyi                        # 查看自己的所有 job
squeue -j 8505666                          # 查看某个 job 状态
scancel 8505666                            # 取消某个 job
tail -f logs/download_8505666.log          # 实时查看日志
```

### Job 状态说明

| 状态 | 含义 |
|------|------|
| `PD` | Pending，排队等待资源 |
| `R`  | Running，正在运行 |
| `CG` | Completing，即将结束 |
| `F`  | Failed，失败 |
| `TO` | Timeout，超时 |

---

## 当前 Batch Jobs

### Jobs #8758293-8758296 — TalkVid 4 并行下载续跑（运行中）

| Job ID | Split | 状态 | 节点 |
|--------|-------|------|------|
| 8758293 | split_0 | R | c0687 |
| 8758294 | split_1 | R | c3004 |
| 8758295 | split_2 | R | c3008 |
| 8758296 | split_3 | R | c3022 |

| 字段 | 值 |
|------|-----|
| 提交时间 | 2026-07-26 15:04 |
| 分区 | short |
| 资源 | 32G 内存，16 CPU × 4 |
| 时间限制 | 48 小时 |
| workers | 4（上轮 8 触发限流，改为 4） |
| sleep | `--sleep-interval 3 --sleep-requests 1`（防限流） |
| 输入 | `/scratch/li.qianyi/TalkVid/data/split_{0-3}.json` |
| 输出 | `/scratch/li.qianyi/TalkVid/clips_download/`（共享，断点续传） |
| 日志 | `/scratch/li.qianyi/TalkVid/logs/download_split{0-3}_<jobid>.log` |
| 脚本 | `/scratch/li.qianyi/TalkVid/run_split_{0-3}.sh` |

查看进度：
```bash
squeue -u li.qianyi
find /scratch/li.qianyi/TalkVid/clips_download/ -name "*.mp4" | wc -l   # 成功下载片段数
wc -l /scratch/li.qianyi/TalkVid/clips_download/logs/failed_urls.txt    # 失败数
tail -n 5 /scratch/li.qianyi/TalkVid/logs/download_split*_*.log         # 所有 split 日志
sacct -u li.qianyi --format=JobID,Elapsed,State,Start,End -S 2026-07-26 # 历史耗时
```

超时后续跑（断点续传）：
```bash
for i in 0 1 2 3; do sbatch /scratch/li.qianyi/TalkVid/run_split_${i}.sh; done
```

---

## 历史 Batch Jobs

| Job ID | 结果 | 问题 | 解决方式 |
|--------|------|------|---------|
| `8499561` | 失败 | yt-dlp 未安装 | `pip install yt-dlp` |
| `8499650` | 失败 | 未传 cookies，bot 检测拦截 | 加 `--cookies` 参数 |
| `8505478` | 取消 | Python 输出缓冲，日志为空 | sbatch 脚本加 `export PYTHONUNBUFFERED=1` |
| `8505513` | 取消 | 格式过严仅允许 mp4，大量视频只有 webm | 改为 `bestvideo+bestaudio/best` |
| `8505666` | 取消 | n-challenge 失败，所有视频只返回缩略图 | 见下方解决记录 |
| `8728615` | 失败 | `CondaError`: sbatch 非交互式 shell 不支持 `conda activate` | sbatch 脚本改用 `/home/li.qianyi/envs/hallo/bin/python` 完整路径 |
| `8728653` | 失败 | `ModuleNotFoundError: No module named 'rich'` | `/home/li.qianyi/envs/hallo/bin/pip install rich` |
| `8728679` | 取消 | n-challenge 仍失败：`download_clips.py` 传了 `--ignore-config`，`~/.config/yt-dlp/config` 被忽略 | 直接修改脚本硬编码 `--js-runtimes`（见下方）|
| `8729004` | 取消 | 单 job 速率 ~18 mp4/min，48h 内跑不完全量 | 改为 4 并行 job（split_0~3） |
| `8729696-8729699` | 完成 | 运行 ~8h，尾部触发 YouTube IP 限流，57,161 个片段下载成功，约 12.4 万失败 | 加 `--sleep-interval 3 --sleep-requests 1`，workers 从 8 降至 4，重提交续跑 |
| `8756461-8756465` | 取消 | workers=2 速度仅 2000/h，预计 62h 超出 48h 限制 | workers 改为 4，重提交（8758293-8758296） |

### n-challenge 最终解决方案

**问题：** `JS runtimes: none`，yt-dlp 无法解 YouTube n-challenge。

**根本原因：**
1. yt-dlp 2026.07.04 的 `_NodeJsRuntime` 自动检测不识别 conda 安装的 node
2. `download_clips.py` 传了 `--ignore-config`，配置文件方案失效

**修复：**
```bash
# 1. conda 安装 node
conda install -c conda-forge nodejs -y

# 2. 直接修改下载脚本（硬编码路径）
sed -i 's|base_cmd = \[\*base_cmd, "-4", "--ignore-config"\]|base_cmd = [*base_cmd, "-4", "--ignore-config", "--js-runtimes", "node:/home/li.qianyi/envs/hallo/bin/node"]|' \
    /scratch/li.qianyi/TalkVid/data_pipeline/0_video_download/download_clips.py
```

---

## 下载完成后的下一步

下载完成后，需要运行 TalkVid 处理 pipeline（人脸检测、音频 embedding 提取等），再转换为 Hallo3 训练格式：

```bash
# 1. 处理 pipeline（待补充）
# 对应目录：/scratch/li.qianyi/TalkVid/data_pipeline/1_video_rough_segmentation/ 等

# 2. 转换为 Hallo3 格式
python scripts/convert_talkvid_to_hallo3.py \
    --clips_flat /scratch/li.qianyi/TalkVid/clips_download \
    --output /scratch/li.qianyi/hallo3_data_full \
    --dataset_name talkvid_full \
    --caption_json /scratch/li.qianyi/TalkVid/data/filtered_video_clips_with_captions.json

# 3. 生成训练索引
python hallo3/extract_meta_info.py \
    -r /scratch/li.qianyi/hallo3_data_full \
    -n talkvid_full
```

详细训练计划见 `TRAINING_PLAN_CN.md`。
