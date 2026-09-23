# 多卡训练计划：新 Audio Encoder (LTX-2 VAE)

> 集群: aicr, B200 节点 (8 GPU/node)
> 单卡验证: 已通过 (sft_s2_verify.yaml, 37 条 TalkVidTestData)
> 训练数据: TalkVid English — **148,223 条** clips, 207.6 小时, 每条 5 秒, 4213 个独立视频

---

## 一、数据概况

```
splits/talkvid_english.json:
  总条数:   148,223 条 clip
  总时长:   207.6 小时
  每条时长: 5 秒
  分辨率:   80.9% 1080p, 16% 4K, 3.1% 其他
  fps:      ~24fps
  来源:     4,213 个 YouTube 视频
```

数据在**硬盘**上，需要搬到集群 scratch 并处理成 Hallo3 格式。

---

## 二、数据处理 Pipeline (在集群上执行)

### 整体流程

```
硬盘上的原始视频
  ↓ Step 1: 拷贝到集群 scratch
  ↓ Step 2: convert_talkvid_to_hallo3.py (视频/人脸/caption)
  ↓ Step 3: extract_audio_emb_ltx_vae.py (音频 VAE 编码)
  ↓ Step 4: extract_meta_info.py (生成 json 索引)
  ↓
准备就绪，开始多卡训练
```

### Step 1: 拷贝数据到集群

```bash
# 在本地/硬盘上，把 TalkVid clips_flat 传到集群
# 需要的目录: videos-crop/, new_face_info/, audios/
rsync -avP /path/to/TalkVid/clips_flat/ \
    li_qiany_neu@aicr:/scratch/li_qiany_neu/TalkVid/clips_flat/

# 同时把 talkvid_english.json 传过去
scp splits/talkvid_english.json \
    li_qiany_neu@aicr:/scratch/li_qiany_neu/TalkVid/
```

预估大小：148K 条 × 5 秒视频 ≈ **300-500 GB**

### Step 2: 转换为 Hallo3 格式

```bash
# 在集群 GPU 节点上
python scripts/convert_talkvid_to_hallo3.py \
    --clips_flat /scratch/li_qiany_neu/TalkVid/clips_flat \
    --output /scratch/li_qiany_neu/hallo3_data_en \
    --dataset_name talkvid_en \
    --num_workers 32
```

这一步会生成：
```
/scratch/li_qiany_neu/hallo3_data_en/
├── videos/        ← 148K 个 .mp4
├── images/        ← 每个视频的抽帧 (可选，extract_meta_info 需要)
├── face_emb/      ← 148K 个 .pt (人脸 embedding)
├── face_mask/     ← 148K 个 .png (人脸 mask)
├── audio_emb/     ← 旧格式，后面会用 VAE 重新提取覆盖
└── caption/       ← 148K 个 .txt
```

预估时间：32 workers, ~4-8 小时

### Step 3: 用 LTX-2 VAE 提取音频 embedding (关键!)

这一步替换旧的 wav2vec embedding 为新的 VAE embedding：

```bash
# 方案 A: 单 GPU (简单但慢)
python scripts/extract_audio_emb_ltx_vae.py \
    --audio_dir /scratch/li_qiany_neu/TalkVid/clips_flat/audios \
    --output_dir /scratch/li_qiany_neu/hallo3_data_en/audio_emb \
    --checkpoint /scratch/li_qiany_neu/pretrained_models/ltx2.5/vae/ltx-2.5-audio-vae-bf16.safetensors

# 方案 B: 4 并行 (推荐，用 SLURM array job)
# 新建 slurm/run_extract_vae_emb.sh:
```

**SLURM 并行提取脚本** `slurm/run_extract_vae_emb.sh`:

```bash
#!/bin/bash
#SBATCH -p b200-batch
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3              # 4 个并行 job
#SBATCH --time=08:00:00
#SBATCH -o /scratch/li_qiany_neu/hallo3_logs/extract_vae_%A_%a.log

module load cuda/13.1.1
conda activate hallo3
cd ~/hallo3

echo "开始 VAE 提取: rank=${SLURM_ARRAY_TASK_ID}/4, $(date)"

python scripts/extract_audio_emb_ltx_vae.py \
    --audio_dir /scratch/li_qiany_neu/TalkVid/clips_flat/audios \
    --output_dir /scratch/li_qiany_neu/hallo3_data_en/audio_emb \
    --checkpoint /scratch/li_qiany_neu/pretrained_models/ltx2.5/vae/ltx-2.5-audio-vae-bf16.safetensors \
    --parallelism 4 \
    --rank "$SLURM_ARRAY_TASK_ID"

echo "✅ rank=${SLURM_ARRAY_TASK_ID} 完成: $(date)"
```

```bash
sbatch slurm/run_extract_vae_emb.sh
```

预估时间：4 GPU 并行, 148K 条 × 5 秒 ≈ **2-4 小时**
输出：148K 个 .pt 文件，每个 (T, 128)，T ≈ 125 (5秒 × 25fps)

### Step 4: 生成训练 json 索引

```bash
python hallo3/extract_meta_info.py \
    -r /scratch/li_qiany_neu/hallo3_data_en \
    -n talkvid_en

# 检查生成了多少条
python -c "import json; d=json.load(open('data/talkvid_en.json')); print(f'{len(d)} 条')"
```

### Step 5: 数据完整性检查

```bash
# 确认各目录文件数一致
echo "videos:    $(ls /scratch/li_qiany_neu/hallo3_data_en/videos/ | wc -l)"
echo "audio_emb: $(ls /scratch/li_qiany_neu/hallo3_data_en/audio_emb/ | wc -l)"
echo "face_emb:  $(ls /scratch/li_qiany_neu/hallo3_data_en/face_emb/ | wc -l)"
echo "face_mask: $(ls /scratch/li_qiany_neu/hallo3_data_en/face_mask/ | wc -l)"
echo "json:      $(python -c \"import json; print(len(json.load(open('data/talkvid_en.json'))))\")"

# 抽查一个 audio_emb 的 shape
python -c "
import torch, glob
f = glob.glob('/scratch/li_qiany_neu/hallo3_data_en/audio_emb/*.pt')[0]
e = torch.load(f, map_location='cpu')
print(f'{f}: shape={e.shape}')  # 期望 (T, 128)
"
```

---

## 三、多卡训练方式

使用**数据并行** (Data Parallel)，每张 GPU 有完整模型副本，各自处理不同数据。

```
8 卡数据并行:
  GPU 0: 处理 batch 0          GPU 4: 处理 batch 4
  GPU 1: 处理 batch 1          GPU 5: 处理 batch 5
  GPU 2: 处理 batch 2          GPU 6: 处理 batch 6
  GPU 3: 处理 batch 3          GPU 7: 处理 batch 7
  
  ↓ 各自 forward
  ↓ 各自 backward
  ↓ all-reduce 同步梯度 (求平均)
  ↓ 一起更新权重
  
  等效 batch_size = 8
```

---

## 四、训练配置 `configs/sft_s2_en_multigpu.yaml`

```yaml
args:
  checkpoint_activations: True
  model_parallel_size: 1
  experiment_name: ltx-vae-english-148k
  mode: finetune

  load: /scratch/li_qiany_neu/pretrained_models/hallo3

  no_load_rng: True
  train_iters: 30000               # 148K 数据, 正式训练 30K 步
  eval_iters: 1
  eval_interval: 1000
  eval_batch_size: 1
  save: /scratch/li_qiany_neu/train_output_en
  save_interval: 1000              # 每 1000 步存 checkpoint
  log_interval: 10
  train_data: [
    "data/talkvid_en.json",        # ← 用英语数据集
  ]
  valid_data: [
    "data/talkvid_en.json",
  ]
  split: 1,0,0
  num_workers: 8
  force_train: True
  only_log_video_latents: True

data:
  target: data_video.Stage2_SFTDataset
  params:
    video_size: [ 480, 720 ]
    fps: 8
    max_num_frames: 49
    skip_frms_num: 3.

deepspeed:
  train_micro_batch_size_per_gpu: 1
  gradient_accumulation_steps: 1
  # 等效 batch_size = 8 GPU × 1 × 1 = 8
  steps_per_print: 10
  gradient_clipping: 0.1
  zero_optimization:
    stage: 2
    cpu_offload: false
    contiguous_gradients: false
    overlap_comm: true
    reduce_scatter: true
    reduce_bucket_size: 1000000000
    allgather_bucket_size: 1000000000
    load_from_fp32_weights: false
  zero_allow_untested_optimizer: true
  bf16:
      enabled: True
  fp16:
      enabled: False
  loss_scale: 0
  loss_scale_window: 400
  hysteresis: 2
  min_loss_scale: 1

  optimizer:
    type: AdamW
    params:
      lr: 1e-5
      betas: [ 0.9, 0.95 ]
      eps: 1e-8
      weight_decay: 1e-4
  activation_checkpointing:
    partition_activations: false
    contiguous_memory_optimization: false
  wall_clock_breakdown: false
```

---

## 五、SLURM 训练脚本 `slurm/run_train_en_8gpu.sh`

```bash
#!/bin/bash
#SBATCH -p b200-batch
#SBATCH --gres=gpu:b200:8              # 整机 8 卡
#SBATCH --mem=512G
#SBATCH --cpus-per-task=64             # 8卡 × 8 workers
#SBATCH --time=24:00:00               # 24 小时
#SBATCH -o /scratch/li_qiany_neu/hallo3_logs/train_en_8gpu_%j.log

HALLO3_DIR="$HOME/hallo3"
mkdir -p /scratch/li_qiany_neu/hallo3_logs

module load cuda/13.1.1
conda activate hallo3

cd "$HALLO3_DIR"

echo "===== 8卡英语训练开始: $(date) ====="
echo "节点: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=8 \
    hallo3/train_video.py \
    --base configs/cogvideox_5b_i2v_s2.yaml configs/sft_s2_en_multigpu.yaml \
    --seed $RANDOM

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ 训练异常退出 (exitcode: $EXIT_CODE)"
    exit $EXIT_CODE
fi

echo "✅ 训练完成: $(date)"
```

---

## 六、完整执行步骤

```
阶段              命令                                       预估时间     在哪里跑
────              ──────                                     ──────      ──────
0. 清除debug      find . -name "*.py" -exec sed -i           1 分钟      集群 login
   print          '/\[AUDIO_DEBUG\]/d' {} +

1. 传数据到集群   rsync clips_flat → scratch                  数小时      本地→集群
                  (取决于网速和数据量)

2. 格式转换       convert_talkvid_to_hallo3.py               4-8 小时    集群 CPU/GPU
                  --num_workers 32

3. VAE 提取       sbatch run_extract_vae_emb.sh              2-4 小时    集群 4×GPU
                  (4 并行 array job)

4. 生成 json      extract_meta_info.py -n talkvid_en         几分钟      集群 login

5. 数据检查       对比各目录文件数                             几分钟      集群 login

6. 多卡验证       srun 2卡 + 跑 2 步                         10 分钟     集群 交互式
                  确认没报错

7. 正式训练       sbatch run_train_en_8gpu.sh                ~24 小时    集群 8×B200
                  30K iters, 8卡
```

---

## 七、训练规模估算

```
数据量:    148,223 条
batch:     8 (8卡 × 1/卡)
1 epoch = 148,223 / 8 = 18,528 步
30K 步 ≈ 1.6 个 epoch

每步耗时:  ~6.3s (单卡实测) → 多卡通信开销 → ~7-8s
30K 步 × 7.5s = 225,000s ≈ 62.5 小时

考虑实际情况（checkpoint 保存、eval 等）:
  预估 ~70-80 小时 ≈ 3 天

24 小时 SLURM job 跑 ~10K 步，需要提交 3 次（或写自动续训脚本）
```

---

## 八、自动续训脚本（可选）

如果 24 小时跑不完，加上自动续训：

```bash
#!/bin/bash
# slurm/run_train_en_8gpu_auto.sh
#SBATCH -p b200-batch
#SBATCH --gres=gpu:b200:8
#SBATCH --mem=512G
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH -o /scratch/li_qiany_neu/hallo3_logs/train_en_8gpu_%j.log

HALLO3_DIR="$HOME/hallo3"
THIS_SCRIPT="$HALLO3_DIR/slurm/run_train_en_8gpu_auto.sh"
CKPT_DIR="/scratch/li_qiany_neu/train_output_en"

mkdir -p /scratch/li_qiany_neu/hallo3_logs

module load cuda/13.1.1
conda activate hallo3
cd "$HALLO3_DIR"

# 自动找最新 checkpoint 续训
LATEST_CKPT=$(ls -d "$CKPT_DIR"/[0-9]* 2>/dev/null | sort -V | tail -1)
if [ -n "$LATEST_CKPT" ]; then
    echo "续训 from: $LATEST_CKPT"
    sed -i "s|load:.*|load: $LATEST_CKPT|" configs/sft_s2_en_multigpu.yaml
fi

echo "===== 训练开始: $(date) ====="

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=8 \
    hallo3/train_video.py \
    --base configs/cogvideox_5b_i2v_s2.yaml configs/sft_s2_en_multigpu.yaml \
    --seed $RANDOM

EXIT_CODE=$?

# 训练完成检查
if [ -d "$CKPT_DIR/30000" ]; then
    echo "🎉 30K 步训练完成!"
else
    echo "未完成，自动重提交..."
    sbatch "$THIS_SCRIPT"
fi
```

---

## 九、监控训练

```bash
# job 状态
squeue -u li_qiany_neu

# 实时日志
tail -f /scratch/li_qiany_neu/hallo3_logs/train_en_8gpu_*.log

# loss 趋势
grep "loss" /scratch/li_qiany_neu/hallo3_logs/train_en_8gpu_*.log | \
    awk '{print NR, $NF}' | tail -50

# checkpoint 列表
ls -lh /scratch/li_qiany_neu/train_output_en/
```

---

## 十、关键参数与调优

### 学习率

```
8 卡 batch_size=8, 基础 lr=1e-5
线性缩放: lr = 1e-5 × 8 = 8e-5 (可能太大)
建议: 先用 1e-5 不动，loss 不降再试 2e-5
```

### 训练步数

| 步数 | epoch 数 | 用途 |
|------|---------|------|
| 1,000 | 0.05 | 快速验证 loss 在下降 |
| 10,000 | 0.54 | 初步结果，可以跑推理看效果 |
| 30,000 | 1.6 | 正式训练 |

### GPU 选择

| GPU 数 | batch | 30K 步耗时 | 适用 |
|--------|-------|-----------|------|
| 4 | 4 | ~5 天 | 保守方案 |
| **8** | **8** | **~3 天** | **推荐** |
| 8 + grad_accum=2 | 16 | ~6 天(步数不变但每步慢) | 更大 batch |

---

## 十一、常见问题

### Q: 数据太多，从硬盘传到集群太慢？
→ 先传一部分（如 1 万条）开始训练，剩下的后台继续传
→ 或者在集群上直接下载（如果有 YouTube 链接）

### Q: 148K 条需要全用吗？
→ 不一定。可以先抽 10K-50K 条训练，看效果再决定是否用全量
→ 抽样方法: `head -10000 data/talkvid_en.json > data/talkvid_en_10k.json`
（注意 json 格式需要用 python 处理，不能直接 head）

### Q: extract_meta_info.py 生成的条数比预期少？
→ 检查哪些文件缺失: face_emb, face_mask, audio_emb, caption 任一缺失都会跳过
→ 跑 `scripts/convert_talkvid_to_hallo3.py` 时有些视频可能处理失败

### Q: 显存不够 (8卡 OOM)?
→ 开启 `cpu_offload: true`
→ 减少 `num_workers` (8→4)
→ 用 `gradient_accumulation_steps: 2` 替代更多 GPU
