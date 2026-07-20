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

### 常用查询命令

```bash
squeue -u li.qianyi                        # 查看自己的所有 job
squeue -j 8499561                          # 查看某个 job 状态
scancel 8499561                            # 取消某个 job
tail -f logs/download_8499650.log          # 实时查看日志
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

### Job #8499561 — TalkVid 全量视频下载

| 字段 | 值 |
|------|-----|
| 提交时间 | 2026-07-20 |
| 分区 | short |
| 资源 | 32G 内存，16 CPU |
| 时间限制 | 48 小时 |
| 日志 | `/scratch/li.qianyi/TalkVid/logs/download_84995650.log` |

**功能：** 从 YouTube 下载 TalkVid 全量数据集（181,752 个视频片段）

**输入：**
```
/scratch/li.qianyi/TalkVid/data/filtered_video_clips_with_captions.json
```

**输出：**
```
/scratch/li.qianyi/TalkVid/clips_download/
├── {VIDEO_ID}/
│   ├── {VIDEO_ID}_{start}_{end}.mp4
│   └── ...
└── json_logs/              ← 断点续传依据，记录已下载的片段
```

**查看进度：**
```bash
squeue -u li.qianyi 
tail -f /scratch/li.qianyi/TalkVid/logs/download_8499650.log

# 查已下载数量
ls /scratch/li.qianyi/TalkVid/clips_download/ | wc -l    
```

**断点续传：** 如果 job 超时或失败，直接重新提交，已下载的自动跳过：
```bash
sbatch /scratch/li.qianyi/TalkVid/run_download.sh
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
