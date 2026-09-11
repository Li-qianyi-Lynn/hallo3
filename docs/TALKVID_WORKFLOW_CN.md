# 使用 TalkVid 数据集运行 Hallo3

本文档介绍在 Slurm 集群上将 TalkVid 数据转换为 Hallo3 格式，并完成微调和推理的完整流程。

---

## 0. 前置准备

### 创建 conda 环境

```bash
# 如果目录已存在但不是 conda 环境，先删除
rm -rf /home/li.qianyi/envs/hallo

# 创建环境
conda create --prefix /home/li.qianyi/envs/hallo python=3.10 -y
conda activate /home/li.qianyi/envs/hallo

# 安装依赖（requirements.txt 中 pyav==14.0.1 有误，先去掉）
grep -v "pyav==14.0.1" requirements.txt > requirements_fixed.txt
pip install -r requirements_fixed.txt
```

### 申请 GPU 节点

```bash
# 查看当前空闲 GPU
sinfo -o "%P %G %N %t" | grep idle

  srun -p sharing --gres=gpu:h100:1 --mem=60G --pty bash                                                     

# 注意：sharing 分区不支持 --time 参数，去掉即可
```

### 验证环境

```bash
conda activate /home/li.qianyi/envs/hallo
python -c "import torch, cv2; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 1. 完整 Pipeline 一览

```
集群路径：/scratch/li.qianyi/TalkVid/

Step 0  ✅ 下载完成
        clips_download/{VIDEO_ID}/{VIDEO_ID}_{start}_{end}.{mp4,m4a}
        脚本：data_pipeline/0_video_download/download_clips.py
        提交：sbatch run_download.sh

Step 1  扁平化结构
        clips_flat/videos/  ← 所有 .mp4 软链接
        clips_flat/audios/  ← 所有 .m4a 软链接
        脚本：data_pipeline/4_flatten_for_training/flatten_clips.py
        提交：sbatch data_pipeline/4_flatten_for_training/run_flatten.sh

Step 2a face_crop（需 GPU，可与 2b 并行）
        clips_flat/videos-crop/
        脚本：src/data_preprocess/scripts/data_process/face_crop.sh

Step 2b extract_audio_emb（需 GPU，可与 2a 并行）
        clips_flat/short_clip_aud_embeds/
        脚本：src/data_preprocess/scripts/data_process/extract_audio_emb.sh

Step 2c extract_face_info（需 2a 完成后才能跑）
        clips_flat/new_face_info/
        脚本：src/data_preprocess/scripts/data_process/extract_face_info.sh

Step 3  格式转换 → Hallo3 训练格式
        /scratch/li.qianyi/hallo3_data/
        脚本：/scratch/li.qianyi/hallo3/scripts/convert_talkvid_to_hallo3.py

Step 4  生成训练索引
        /scratch/li.qianyi/hallo3/data/talkvid.json
        脚本：/scratch/li.qianyi/hallo3/hallo3/extract_meta_info.py

Step 5  Fine-tune Hallo3
        脚本：scripts/finetune_multi_gpus_s1.sh → s2.sh
```

---

## 2. 数据全流程概览

### 原始数据：JSON 元数据

`filtered_video_clips_with_captions.json` 里每条记录对应一个说话人片段：

```json
{
  "video_url": "https://www.youtube.com/watch?v=xxx",
  "start": 10.5,
  "end": 16.2,
  "caption": "A person talking about..."
}
```

共 181,752 条记录。

---

### 第一步：下载 → `clips_download/`

**实际路径：** `/scratch/li.qianyi/TalkVid/clips_download/`

```bash
python download_clips.py \
    --input /scratch/li.qianyi/TalkVid/data/filtered_video_clips_with_captions.json \
    --output /scratch/li.qianyi/TalkVid/clips_download/ \
    --cookies /scratch/li.qianyi/TalkVid/youtube_cookies.txt
```

按 JSON 里的时间段从 YouTube 裁剪下载，得到：

```
/scratch/li.qianyi/TalkVid/clips_download/
└── {VIDEO_ID}/
    └── {VIDEO_ID}_{start}_{end}.mp4   ← 几秒钟的说话人原始片段
```

> **当前状态：** 下载卡在 yt-dlp n-challenge 问题，11,574 个 URL 失败，详见 `SLURM_JOBS_CN.md`。

---

### 第二步：TalkVid pipeline → `clips_flat/`

**实际路径：** `/scratch/li.qianyi/TalkVid/clips_flat/`

对原始片段做人脸检测、裁剪、音频特征提取（脚本在 `data_pipeline/` 下）：

```
/scratch/li.qianyi/TalkVid/clips_flat/
├── videos-crop/*.mp4              ← 人脸区域裁剪后的视频
├── new_face_info/*.pt             ← 每帧人脸检测结果
│                                  #   list[帧] → list[人脸] → {bbox, 关键点, embedding}
└── short_clip_aud_embeds/*.pt     ← 音频 embedding
                                   #   dict{'global_embeds': Tensor[T, 1, 768]}
```

---

### 第三步：格式转换 → `hallo3_data/`（Hallo3 训练格式）

**实际路径：** `/scratch/li.qianyi/hallo3_data/`

```bash
python scripts/convert_talkvid_to_hallo3.py \
    --clips_flat /scratch/li.qianyi/TalkVid/clips_flat/ \
    --output /scratch/li.qianyi/hallo3_data/
```

```
/scratch/li.qianyi/hallo3_data/
├── videos/*.mp4                   ← 同 videos-crop
├── images/{stem}/*.jpg            ← 逐帧拆出的图片
├── face_emb/*.pt                  ← 人脸 embedding，所有帧平均 → Tensor[512]
├── face_mask/*.png                ← 人脸区域二值 mask
├── audio_emb/*.pt                 ← 音频 embedding Tensor[T, 1, 768]
└── caption/*.txt                  ← "A person talking."
```

> **当前状态：** 已有 42 个测试样本转换完成，可直接用于训练流程验证。

---

**本质：** 每条训练样本 = 一段说话人短视频 + 人脸位置/特征 + 音频特征，
用于训练"给定音频 → 生成对应嘴型和头部运动"的模型。

---

## 1. TalkVid 数据结构说明

TalkVid 预处理后的数据存放在 `clips_flat/`：

```
/scratch/li.qianyi/TalkVid/clips_flat/
├── videos-crop/            # 裁剪后的人脸视频 (.mp4)
├── new_face_info/          # 每帧人脸检测结果 (.pt)
│                           #   list[帧] → list[人脸] → {bbox, 关键点, embedding}
└── short_clip_aud_embeds/  # 音频 embedding (.pt)
                            #   dict{'global_embeds': Tensor[T, 1, 768]}
```

检查各目录文件数量（理论上应一致）：
```bash
cd /scratch/li.qianyi/TalkVid/clips_flat
ls videos-crop | wc -l            # 45
ls new_face_info | wc -l          # 42（缺 3 个，测试阶段可忽略）
ls short_clip_aud_embeds | wc -l  # 45
```

查找缺失的人脸信息文件：
```bash
comm -23 <(ls videos-crop | sed 's/\.[^.]*$//' | sort) \
         <(ls new_face_info | sed 's/\.[^.]*$//' | sort)
```

---

## 2. 数据转换：TalkVid → Hallo3 格式


python scripts/convert_talkvid_to_hallo3.py --clips_flat /scratch/li.qianyi/TalkVid/clips_flat --output /scratch/li.qianyi/hallo3_data --dataset_name talkvid
```

脚本对每个视频做的事情：

| 步骤 | 输入 | 输出 |
|------|------|------|
| 复制视频 | `videos-crop/*.mp4` | `videos/*.mp4` |
| 抽取帧 | `videos-crop/*.mp4` | `images/*/000000.jpg ...` |
| 人脸 embedding | `new_face_info/*.pt` → 对所有帧取平均 | `face_emb/*.pt`（Tensor[512]） |
| 人脸 mask | `new_face_info/*.pt` → bbox → 生成二值图 | `face_mask/*.png` |
| 音频 embedding | `short_clip_aud_embeds/*.pt` → 提取 tensor | `audio_emb/*.pt`（Tensor[T,1,768]） |
| 文字描述 | — | `caption/*.txt`（"A person talking."） |

转换后输出目录结构：
```
/scratch/li.qianyi/hallo3_data/
├── videos/
├── images/
├── face_emb/
├── face_mask/
├── audio_emb/
└── caption/
```

---

## 3. 生成训练索引文件（Meta JSON）

```bash
cd /hallo3

python hallo3/extract_meta_info.py \
    -r /scratch/li.qianyi/hallo3_data \
    -n talkvid
```

生成 `data/talkvid.json`，这是 Hallo3 训练时读取的数据索引。

验证生成结果：
```bash
python -c "import json; d=json.load(open('data/talkvid.json')); print(f'共 {len(d)} 条样本')"
```

---

## 4. 微调（Fine-tuning）

编辑 `configs/sft_s1.yaml`，填入数据路径：

```yaml
train_data: [
  "./data/talkvid.json",
]
valid_data: [
  "./data/talkvid.json",
]
```

运行 Stage 1：
```bash
bash scripts/finetune_multi_gpus_s1.sh
```

Stage 1 完成后运行 Stage 2：
```bash
bash scripts/finetune_multi_gpus_s2.sh
```

checkpoint 每 500 步自动保存到 `./stage-1/` 和 `./stage-2/`。

---

## 训练计划

### 阶段一：Smoke Test（当前）

**目标**：验证数据格式正确、训练流程跑通、loss 正常下降，不追求生成质量。

| 参数 | 值 |
|------|----|
| 配置文件 | `configs/sft_talkvid.yaml` |
| SLURM 脚本 | `slurm/run_finetune_s1.sh` |
| 迭代数 | 500 |
| 视频尺寸 | 320×512（降低显存占用） |
| 最大帧数 | 25（降低显存占用） |
| GPU | 8×A6000（d3232，sharing 分区） |
| DeepSpeed | ZeRO-2，无 cpu_offload |
| checkpoint | 每 25 步保存，支持自动续训 |

**遇到的 OOM 问题及解法**：

| 尝试 | GPU | 问题 | 原因 |
|------|-----|------|------|
| 4×L40S | 44.5GB/卡 | CUDA OOM | 显存不足 |
| 8×A6000 + cpu_offload | 48GB/卡 | SIGKILL (-9) | CPU 内存溢出 |
| 8×A6000 + ZeRO-3 | 48GB/卡 | TypeError | ZeRO-3 与 ColumnParallelLinear 不兼容 |
| 8×A6000 + ZeRO-2 + 原始尺寸 | 48GB/卡 | SIGKILL (-9) | 49帧×480×720 activation 超显存 |
| 1×H200 + xformers | 80GB/卡 | **进行中** | ZeRO-2 单卡，内存高效注意力 |

**遇到的环境兼容性问题及解法**：

| 报错 | 原因 | 解决方法 |
|------|------|---------|
| `RuntimeError: NVIDIA driver too old (found version 12080)` | 集群所有节点（含 H200）CUDA 驱动为 12.8，PyTorch cu130 要求 ≥13.0 | 重装 PyTorch：`pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124` |
| `ERROR: Could not find a version xformers==0.0.27.post2` | cu124 索引中无该版本 | 改用 `xformers==0.0.28.post1 --index-url https://download.pytorch.org/whl/cu124` |
| `ncclUnhandledCudaError: Call to CUDA function failed` (NCCL 2.29.7) | torchrun 单卡仍初始化 NCCL，P2P/InfiniBand 调用在部分节点上失败 | 训练命令加 `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1` |
| `can't divide axis of length 640 in chunks of 1350` (einops) | 模型空间 token 数硬编码为 1350（对应 480×720），视频尺寸不可改 | 必须用 480×720，不能降低分辨率 |
| `shape '[4, 48, 4, 64]' is invalid` (reshape) | 模型时序维度硬编码为 49 帧，帧数不可改 | 必须用 49 帧，不能减少帧数 |
| `CUDA illegal memory access` (单卡无 xformers) | 标准 O(n²) 注意力对 17550 token 约需 29GB/层，超出显存 | 安装 xformers（memory-efficient attention） |

---

### 阶段二：正式训练（Smoke Test 跑通后）

**目标**：用完整分辨率和更多迭代数训练，追求生成质量。

| 参数 | 值 |
|------|----|
| 迭代数 | 2000～5000 |
| 视频尺寸 | 480×720（恢复原始） |
| 最大帧数 | 49（恢复原始） |
| GPU | 8×H200（需等节点空闲） |
| DeepSpeed | ZeRO-2 |

```bash
# 正式训练时修改配置
# configs/sft_talkvid.yaml:
#   train_iters: 2000
#   video_size: [480, 720]
#   max_num_frames: 49
#   GPU: --gres=gpu:h200:8
```

---

## 5. 推理（Inference）

用我自己的图
### 运行推理

```bash
bash scripts/inference_long_batch.sh \
    my_inference/input_talkvid.txt \
    my_inference/outputs/
```

生成视频保存在 `my_inference/outputs/`。

---

## 常见报错解决

| 报错 | 原因 | 解决方法 |
|------|------|---------|
| `pyav==14.0.1` 安装失败 | PyPI 上没有这个包 | 用 `requirements_fixed.txt` 安装 |
| `Killed`（加载 .pt 文件时） | 登录节点内存限制 | 在计算节点上运行 |
| `prefix already exists` | conda 环境目录已存在 | `rm -rf /home/li.qianyi/envs/hallo` 后重建 |
| `DirectoryNotACondaEnvironmentError` | 目录存在但不是 conda 环境 | 同上，先删除目录 |
| `srun: Requested time limit is invalid` | sharing 分区不支持 --time | 去掉 `--time` 参数 |
| new_face_info 少 3 个文件 | 对应视频人脸检测失败 | 测试阶段可忽略，42/45 正常 |
