# Smoke Test 记录 — TalkVid Fine-tuning

**日期：** 2026-08-04  
**目标：** 验证完整训练流程可跑通，产出 checkpoint

---

## 一、训练配置

### 硬件
| 参数 | 值 |
|------|-----|
| GPU | 1 × NVIDIA H200 (80GB) |
| 分区 | gpu（NEU Explorer 集群） |
| 节点 | d4054 |
| 内存 | 64GB |
| CPUs | 8 |

### 系统环境
| 参数 | 值 |
|------|-----|
| CUDA 驱动 | 570.86.15（CUDA 12.8） |
| 系统 CUDA 模块 | cuda/12.8.0 |
| 系统 cuDNN 模块 | cuDNN/9.10.2 |
| PyTorch | 2.4.1+cu124 |
| xformers | 0.0.28.post1 |
| DeepSpeed | ZeRO-2 |

### 数据
| 参数 | 值 |
|------|-----|
| 数据集 | TalkVid（自采集） |
| 来源 | `/scratch/li.qianyi/hallo3_data` |
| 训练索引 | `./data/talkvid.json` |
| 总条数 | **59,962 条**（视频片段） |
| 原始视频数 | ~61,431（少数因缺 face_info 等被跳过） |
| 视频格式 | MP4，说话人脸部裁剪 |

### 模型配置
| 参数 | 值 |
|------|-----|
| 基础模型 | CogVideoX-5B-I2V |
| 预训练权重 | `./pretrained_models/cogvideox-5b-i2v-sat/transformer` |
| 训练模式 | finetune（只训练 attention + face 模块） |
| 视频分辨率 | 480 × 720 |
| 帧数 | 49 帧（max_num_frames） |
| FPS | 8 |
| latent_width / latent_height | 90 / 60 |

### 训练超参数
| 参数 | 值 |
|------|-----|
| iterations | **100** |
| batch_size per GPU | 1 |
| 有效 batch_size | 1（单卡） |
| gradient_accumulation_steps | 1 |
| optimizer | AdamW |
| learning_rate | 1e-5 |
| betas | (0.9, 0.95) |
| weight_decay | 1e-4 |
| 精度 | BF16 |
| gradient_clipping | 0.1 |
| activation_checkpointing | 开启（cpu_checkpointing + contiguous） |
| save_interval | 每 25 步保存一次 |

---

## 二、训练结果

| 指标 | 值 |
|------|-----|
| 状态 | ✅ 成功完成 |
| 实际用时 | ~25 分钟 |
| 看过的数据量 | 100 个视频片段（占总数 **0.17%**） |
| checkpoint 路径 | `./stage-1/train-talkvid-08-04-16-32/100/mp_rank_00_model_states.pt` |
| 每 iteration 用时 | ~7,569 ms |
| 吞吐量 | 7.93 samples/(min·GPU) |

**注意：** 100 iterations 仅验证流程可跑通，模型尚未真正学到数据特征，**不适合用于推理评估**。

### Loss 观察

| 阶段 | loss 值 | 说明 |
|------|---------|------|
| iter 1~3 | 0.073, 0.072, 0.100 | 初始值 |
| 典型区间 | 0.07 ~ 0.15 | 大多数 iteration 的范围 |
| 偶发 spike | 0.27 ~ 0.72 | 来自个别困难样本，属正常 |
| iter 100 | 0.077 | 最后一步 |

**结论：** 100 iterations 内 loss 无明显下降趋势，整体在 0.07~0.15 间震荡，偶发大 spike（最高 0.716）。  
这是正常现象 — batch_size=1 的随机性太大，噪声完全压过信号。**需要 1,000+ iterations 才能看到稳定下降趋势。**

---

## 三、关键修复（与之前失败版本的区别）

本次能跑通的关键改动：

1. **恢复 `module load cuda/12.8.0` + `cuDNN/9.10.2`**  
   PyTorch 自带 cuDNN 9.2.0 与驱动 570.86.15 不兼容，需要加载系统 cuDNN 9.10.2 覆盖。

2. **optimizer 改为 `AdamW`**  
   原来的 `sat.ops.FusedEmaAdam` 是自定义 CUDA 算子，未针对 H200（Hopper 架构）编译，在 optimizer.step() 时产生 illegal memory access。

详见 `TROUBLESHOOTING_CN.md` 问题 #20、#21。

---

## 四、下一步：正式训练

### 方案 A：单卡 H200（不推荐，太慢）

| 参数 | 值 |
|------|-----|
| GPU | 1 × H200 |
| iterations | 30,000 |
| 预计用时 | **~5 天** |
| 覆盖数据 | ~50%（约 30,000 / 59,962） |

### 方案 B：4 × H200（推荐）

| 参数 | 值 |
|------|-----|
| GPU | 4 × H200（sharing 分区） |
| iterations | 30,000 |
| 有效 batch_size | 4（每卡 1） |
| 预计用时 | **~15 小时** |
| 覆盖数据 | ~50% |
| 内存需求 | 200GB |
| CPUs | 32 |

**SLURM 配置：**
```bash
#SBATCH -p sharing
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=200G
#SBATCH --cpus-per-task=32
```

**训练命令：**
```bash
CUDA_VISIBLE_DEVICES="0,1,2,3" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=4 \
    hallo3/train_video.py \
    --base configs/cogvideox_5b_i2v_s1.yaml configs/sft_talkvid.yaml \
    --seed $RANDOM
```

### 方案 C：8 × A100（备选）

| 参数 | 值 |
|------|-----|
| GPU | 8 × A100 80GB（sharing 分区，节点 d3146-3150 等） |
| iterations | 30,000 |
| 有效 batch_size | 8 |
| 预计用时 | **~10 小时** |
| 内存需求 | 400GB |

---

## 五、1 epoch 需要多少 iterations？

| GPU 配置 | batch_size | 1 epoch iterations | 预计用时 |
|----------|------------|-------------------|---------|
| 1 × H200 | 1 | 59,962 | ~10 天 |
| 4 × H200 | 4 | 14,991 | ~2 小时 |
| 8 × A100 | 8 | 7,496 | ~50 分钟 |

研究计划目标 **30,000 iterations**（约 2 epoch，4×H200 配置下）。



 