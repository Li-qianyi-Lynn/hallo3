# 2026-10-29：G1 实验执行计划

承接 [2026-10-26_omnilingual_conversion.md](2026-10-26_omnilingual_conversion.md) 和 [research plan](../research_plan/2026-10-23_rp.md)。

---

## 总览

```
已完成 ✅
  ├── Omnilingual wav2vec2 300M 下载 + fairseq2→HF 转换 (420/422 keys)
  ├── AudioProjModel adapter 代码改动 (transformer.py)
  ├── audio_processor.py 取层逻辑改动（最后 12/24 层）
  ├── B0 val200 真值 SyncNet 打分 (conf=7.055)
  └── 训练 JSON 生成脚本 (make_train_json_from_8k.py)

待做（按顺序）
  Step 1: 生成训练 JSON (2700 条)
  Step 2: 预处理训练数据 (face_emb, face_mask, images)
  Step 3: 提取 Omnilingual audio embedding
  Step 4: 更新训练配置，启动 G1 训练
  Step 5: G1 推理 + SyncNet 打分
  并行: B1 原版 Hallo3 val200 推理 + 打分
```

---

## Step 1: 生成训练 JSON

**目的**：从 8K 集剔除 val+eval，随机抽 2700 条，生成训练 JSON。

**前置**：集群上 `scripts/make_train_json_from_8k.py` 需要更新（检查 video 而非 face_emb）。

```bash
cd ~/hallo3
git pull origin train8

python scripts/make_train_json_from_8k.py \
    --splits-8k ~/hallo3/splits/talkvid_en_8k_quality.json \
    --splits-val ~/hallo3/splits/talkvid_val.json \
    --splits-eval ~/hallo3/splits/talkvid_eval.json \
    --data-dir /scratch/li_qiany_neu/talkvid_8k \
    --n 2700 --seed 42 \
    --output ~/hallo3/data/talkvid_train_2700.json
```

**预期输出**：`Wrote 2700 entries to .../talkvid_train_2700.json`

**验证**：
```bash
python -c "import json; data=json.load(open('data/talkvid_train_2700.json')); print(len(data)); print(data[0]['video_path'])"
```

---

## Step 2: 预处理训练数据

**目的**：对 2700 条视频提取 images、face_emb、face_mask。audio_emb 不在这步做（Step 3 用 Omnilingual 单独提取）。

**需要 GPU 节点**（insightface 人脸检测用 GPU）。

```bash
srun --partition=rtx-devel --nodes=1 --gpus=1 --cpus-per-task=8 --mem=128G --time=02:00:00 --account=p2026_0014_neu --pty bash

module load cuda/13.1.1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
conda activate hallo3
cd ~/hallo3/hallo3
```

用 `scripts/process_4k_data.py` 处理（跳过 audio_emb 步骤）：

```bash
python ../scripts/process_4k_data.py \
    --json ../data/talkvid_train_2700.json \
    --data-dir /scratch/li_qiany_neu/talkvid_8k \
    --skip_steps audio_emb
```

> **注意**：如果 `process_4k_data.py` 不支持 `--json` 输入或 `--skip_steps`，需要改用其他方式。
> 备选方案：直接用 `data_preprocess.py` 逐条处理，或写一个只提取 face_emb + images 的脚本。

**预期**：
- `/scratch/li_qiany_neu/talkvid_8k/images/<stem>/000000.jpg` — 2700 条
- `/scratch/li_qiany_neu/talkvid_8k/face_emb/<stem>.pt` — 2700 条
- `/scratch/li_qiany_neu/talkvid_8k/face_mask/<stem>.png` — 2700 条

**验证**：
```bash
ls /scratch/li_qiany_neu/talkvid_8k/face_emb/ | wc -l  # 应为 641 + 2700 = 3341
ls /scratch/li_qiany_neu/talkvid_8k/images/ | wc -l     # 应为 641 + 2700 = 3341
```

**预估耗时**：~1-2 小时（参照之前 2719 条用了 ~80 分钟）。

---

## Step 3: 提取 Omnilingual audio embedding

**目的**：用 Omnilingual wav2vec2 对 2700 条音频提取 embedding，shape `(T, 12, 1024)`。

**需要 GPU 节点**。

```bash
cd ~/hallo3/hallo3

python ../scripts/re_extract_audio_emb.py \
    --json ../data/talkvid_train_2700.json \
    --encoder /scratch/li_qiany_neu/pretrained_models/omnilingual-w2v2-300m \
    --output-dir /scratch/li_qiany_neu/talkvid_8k/audio_emb_omnilingual \
    --output-json ../data/talkvid_train_2700_omnilingual.json
```

**预期输出**：
- `/scratch/li_qiany_neu/talkvid_8k/audio_emb_omnilingual/<stem>.pt` — 2700 个文件
- `data/talkvid_train_2700_omnilingual.json` — 路径更新后的 JSON
- 每个 .pt 文件 shape: `(T, 12, 1024)`

**验证**：
```bash
ls /scratch/li_qiany_neu/talkvid_8k/audio_emb_omnilingual/ | wc -l  # 应为 2700
python -c "
import torch
emb = torch.load('/scratch/li_qiany_neu/talkvid_8k/audio_emb_omnilingual/$(ls /scratch/li_qiany_neu/talkvid_8k/audio_emb_omnilingual/ | head -1)', map_location='cpu')
print(f'shape: {emb.shape}')  # 期望 (T, 12, 1024)
"
```

**支持断点续跑**：如果中断，重新跑同一命令，已存在的 .pt 会自动跳过。

**预估耗时**：~30-60 分钟（Omnilingual 315M 参数，GPU 推理很快）。

---

## Step 4: 更新训练配置，启动 G1 训练

### 4.1 更新训练 YAML

修改 `configs/sft_4k_train.yaml`（或新建 `configs/sft_g1_train.yaml`）：

```yaml
args:
  load: /scratch/li_qiany_neu/pretrained_models/hallo3   # 原版 Hallo3 预训练权重
  train_iters: 7000
  save: /scratch/li_qiany_neu/train_output_g1
  save_interval: 1000    # 每 1000 步存 checkpoint，便于 A3 趋势分析
  train_data: ["../data/talkvid_train_2700_omnilingual.json"]
```

### 4.2 确认 Hallo3 预训练权重

```bash
ls /scratch/li_qiany_neu/pretrained_models/hallo3/1/mp_rank_00_model_states.pt
cat /scratch/li_qiany_neu/pretrained_models/hallo3/latest  # 应输出 1
```

> 如果权重还在下载中，等下载完再开始。

### 4.3 启动训练

```bash
srun --partition=rtx-devel --nodes=1 --gpus=2 --cpus-per-task=16 --mem=256G --time=02:00:00 --account=p2026_0014_neu --pty bash

module load cuda/13.1.1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
conda activate hallo3
cd ~/hallo3/hallo3

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=2 \
    train_video.py \
    --base ../configs/cogvideox_5b_i2v_s2.yaml ../configs/sft_g1_train.yaml \
    --seed 42 2>&1 | tee /scratch/li_qiany_neu/hallo3_logs/train_g1_$(date +%m-%d-%H-%M).log
```

**预期 log**：
- `Missing keys: ['audio_proj.adapter.weight', 'audio_proj.adapter.bias']` — 正常，adapter 随机初始化
- 不应出现 `Skipped shape-mismatched keys` — proj1 shape 完全匹配
- loss 应从 ~0.04 开始下降

**预估耗时**：7000 步 × ~9s/iter ÷ 2 GPU ≈ ~9 小时。用 rtx-devel（限 2h）需要多次续训，或改用 rtx-batch（24h）。

---

## Step 5: G1 推理 + SyncNet 打分

### 5.1 val5 快速验证

```bash
cd ~/hallo3/hallo3

TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 \
python sample_video.py \
    --base ../configs/cogvideox_5b_i2v_s2.yaml ../configs/inference.yaml \
    --seed 42 \
    --wav2vec_model_path /scratch/li_qiany_neu/pretrained_models/omnilingual-w2v2-300m \
    --input-file ../my_inference/input_val5.txt \
    --output-dir /scratch/li_qiany_neu/infer_out/g1_val5
```

> **注意**：不传 `--vae_checkpoint`，走 wav2vec 分支。
> 需要在 inference.yaml 中设置 `load:` 指向 G1 训练的 checkpoint 路径。

### 5.2 SyncNet 打分

```bash
conda activate syncnet
cd ~/syncnet_python
OUT=/scratch/li_qiany_neu/syncnet_out

for dir in /scratch/li_qiany_neu/infer_out/g1_val5/000000-*-seed_42; do
  stem=$(basename "$dir" | sed 's/^000000-//; s/-seed_42$//')
  vid="$dir/000000_with_audio.mp4"
  [ -f "$vid" ] || continue
  echo "=== g1_$stem"
  python run_pipeline.py --videofile "$vid" --reference "g1_$stem" --data_dir "$OUT" </dev/null
  python run_syncnet.py  --videofile "$vid" --reference "g1_$stem" --data_dir "$OUT" </dev/null
done 2>&1 | tee $OUT/g1_val5.log | grep -E "^===|AV offset|Min dist|Confidence|Error"
```

### 5.3 汇总

```bash
grep "Confidence:" $OUT/g1_val5.log | awk '{print $NF}' | awk '{s+=$1; n++} END {printf "G1 val5 Confidence avg: %.3f (n=%d)\n", s/n, n}'
grep "Min dist:" $OUT/g1_val5.log | awk '{print $NF}' | awk '{s+=$1; n++} END {printf "G1 val5 Min dist avg:   %.3f (n=%d)\n", s/n, n}'
```

**成功标准**（见 research plan 第十节）：

| 层级 | 标准 |
|------|------|
| 最低 | conf > B2（噪声基线） |
| 及格 | conf > 3 |
| 良好 | conf > 5 |
| 优秀 | conf ≈ 6.33（B1 水平） |

---

## 并行任务：B1 val200 推理

与上述步骤独立，在 `~/hallo3_orig` 目录跑：

```bash
cd ~/hallo3_orig/hallo3

TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 \
python sample_video.py \
    --base ../configs/cogvideox_5b_i2v_s2.yaml ../configs/inference.yaml \
    --seed 42 \
    --input-file ~/hallo3/my_inference/input_val200.txt \
    --output-dir /scratch/li_qiany_neu/infer_out/orig_hallo3_val200
```

**前置**：Hallo3 预训练权重 + T5 权重 + `pretrained_models` 软链就绪。

**预估耗时**：200 条 × ~10 min = ~33 小时。建议 sbatch 提交。

---

## 依赖关系

```
Step 1 (JSON生成) ─→ Step 2 (预处理) ─→ Step 4 (训练)
                  └─→ Step 3 (audio emb) ──┘       └─→ Step 5 (推理+打分)

B1 推理（独立，可并行）
```

Step 2 和 Step 3 可以在同一个 GPU 节点上**顺序**跑（先 face_emb，再 audio_emb），也可以分两个节点**并行**跑（audio_emb 不依赖 face_emb）。
