# Audio Encoder 替换进度

更新日期：2026-09-11

---

## 总体进度

| Step | 任务 | 状态 |
|------|------|------|
| 1 | 下载 LTX-2 音频 VAE 权重 | 完成 |
| 2 | 将 LTX-2 audio_vae 模块搬进 Hallo3 | 完成 |
| 3 | 修改 Hallo3 pipeline 文件 | 完成 |
| 4 | 重新提取 audio embeddings | 完成 |
| 5 | 小规模验证 (100 iterations) | 完成 |
| 6 | 消融实验 | 未开始 |

---

## Step 1: 下载 LTX-2 VAE 权重 — 完成

- 权重位置：`/scratch/li_qiany_neu/pretrained_models/ltx2.5/vae/ltx-2.5-audio-vae-bf16.safetensors`
- 来源：HuggingFace `Lightricks/LTX-2.5`（需要申请访问权限 + `huggingface-cli login`）

## Step 2: 搬运 LTX-2 audio_vae 模块 — 完成

11 个文件从 LTX-2 搬到 `hallo3/sgm/models/ltx_audio_vae/`：

```
__init__.py, audio_vae.py, ops.py, patchifier.py, types.py,
attention.py, resnet.py, causal_conv_2d.py, causality_axis.py,
downsample.py, normalization.py
```

修改内容：
- 所有 import 从 `from ltx_core.xxx` 改为相对导入 `from .xxx`
- 移除 AudioDecoder、Vocoder、Upsample、Disposable（只需 encoder）

### 独立验证 — 完成

运行 `scripts/verify_ltx_audio_vae.py`，结果：

```
Encoder loaded:       OK (46/46 keys matched)
Forward pass:         OK
Output shape:         (1, 8, 251, 16) → patchified (1, 251, 128)
Temporal fps:         25.1 (target: 25.0)
AudioProjModel input: 640 (was 46080, ratio: 72.0x)
```

关键发现：
- `mid_block_add_attention=False`（checkpoint 里无 mid attention 权重）
- 必须传枚举值 `NormType.PIXEL` / `CausalityAxis.HEIGHT`，不能传字符串

## Step 3: 修改 Hallo3 pipeline 文件 — 完成

共修改 5 个文件：

### 3.1 `hallo3/sgm/models/transformer.py` — AudioProjModel 维度

```python
# 旧默认值:
blocks=12, channels=768  → input_dim = 5×12×768 = 46,080

# 新默认值:
blocks=1, channels=128   → input_dim = 5×1×128 = 640
```

### 3.2 `hallo3/data_video.py` — 训练数据加载（两处）

添加自动兼容逻辑 + `weights_only=False`（PyTorch 2.7 安全限制）：
```python
# LTX-2 VAE: (frames, 5, 128) → unsqueeze → (frames, 5, 1, 128)
if audio_tensor.dim() == 3:
    audio_tensor = audio_tensor.unsqueeze(2)

# PyTorch 2.7: torch.load 需要 weights_only=False
face_emb = torch.load(face_emb_path, weights_only=False)
audio_emb = torch.load(audio_emb_path, weights_only=False)
```

### 3.3 `hallo3/sample_video.py` — 推理 process_audio_emb

同样添加 unsqueeze 逻辑，兼容新旧格式。

### 3.4 `hallo3/train_video.py` — Checkpoint 加载兼容

添加 monkey-patch，在加载预训练 checkpoint 时自动跳过 shape 不匹配的 key：
```
预训练 checkpoint: audio_proj.proj1.weight = (512, 46080)  ← wav2vec 维度
新模型:           audio_proj.proj1.weight = (512, 640)     ← LTX-2 VAE 维度
→ 自动跳过，AudioProjModel 用随机初始化权重重新训练
```

其他所有模块（DiT、text encoder、VAE、face_proj）的权重正常加载。

### 3.5 新增脚本和配置

| 文件 | 用途 |
|------|------|
| `scripts/extract_audio_emb_ltx_vae.py` | 用 LTX-2 VAE 从音频提取 embeddings (T, 128) |
| `scripts/verify_ltx_audio_vae.py` | 独立验证 AudioEncoder 加载和 forward pass |
| `scripts/preprocess_and_extract.sh` | 一键预处理 SLURM 脚本 |
| `scripts/verify_train_100iter.sh` | 100 iteration 验证 SLURM 脚本 |
| `configs/sft_s2_verify.yaml` | 验证用训练配置（100 iter, 1 GPU, Adam optimizer） |

## Step 4: 重新提取 audio embeddings — 完成

### 数据准备
- 45 个 pilot 视频从旧集群传到新集群
- 位置：`/scratch/li_qiany_neu/TalkVidTestData/clips_flat/`
- `data_preprocess.py` 提取了 face_emb、face_mask、audios（37 个视频成功，8 个人脸检测失败跳过）
- `extract_audio_emb_ltx_vae.py` 提取了 LTX-2 VAE audio embeddings（45/45 成功，2 秒完成）

### 元数据生成
- `extract_meta_info.py` 生成 `data/talkvid.json`（37 条有效记录）
- 需要手动修复的问题：`torch.load` 需要 `weights_only=False`

### 输出目录结构
```
/scratch/li_qiany_neu/TalkVidTestData/
├── clips_flat/          # 45 个 mp4 视频（原始）
├── videos -> clips_flat # 软链接（extract_meta_info 需要）
├── audios/              # 从 mp4 提取的 .wav
├── images/              # 视频帧（data_preprocess 生成）
├── face_emb/            # 人脸特征 .pt
├── face_mask/           # 人脸 mask .png
├── audio_emb/           # LTX-2 VAE embeddings .pt, shape (T, 128)
└── caption/             # "A person is talking." placeholder
```

### Audio embedding 验证
```
shape: (1822, 128)  ← 某个视频的示例
含义: 1822 帧 × 128 维 ≈ 72.9 秒 × 25fps
```

## Step 5: 小规模验证 — 运行中

### 训练配置
- 基础配置: `configs/cogvideox_5b_i2v_s2.yaml` (add_audio_module: True)
- 验证配置: `configs/sft_s2_verify.yaml` (100 iter, Adam optimizer)
- 预训练 checkpoint: `/scratch/li_qiany_neu/pretrained_models/hallo3/`
- 数据: `data/talkvid.json` (37 条)
- GPU: 1× RTX PRO 6000 Blackwell, 128GB 内存

### 训练命令
```bash
cd ~/hallo3/hallo3
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=1 \
  train_video.py \
  --base ../configs/cogvideox_5b_i2v_s2.yaml ../configs/sft_s2_verify.yaml \
  --seed 42
```

### 结果 — 通过

| 指标 | 值 |
|------|-----|
| 初始 loss (iter 1) | 4.829E-02 |
| 最终 loss (iter 100) | 4.000E-02 |
| loss 下降幅度 | 17% |
| 速度 | ~9s/iter, 6.7 samples/min/GPU |
| 内存 peak | 76GB allocated, 87GB cached |
| Forward/Backward | 全部正常，无报错 |

结论：LTX-2 VAE audio encoder 替换成功，pipeline 完整跑通。

## Step 6: 消融实验 — 未开始

### 实验设计

| # | 编码器 | 训练数据 | 目的 | 分支 |
|---|--------|----------|------|------|
| A | Wav2Vec2 (原) | 英语为主 (TalkVid pilot) | Baseline | main |
| B | LTX-2 VAE (新) | 英语为主 (TalkVid pilot) | 隔离编码器贡献 | test5/encoder |
| C | Wav2Vec2 (原) | 平衡多语言 | 隔离数据贡献 | 待建 |
| D | LTX-2 VAE (新) | 平衡多语言 | 完整方案 | 待建 |

### 评估指标

- **Lip-sync 精度**: SyncNet confidence score（英语 + 非英语分别评估）
- **视频质量**: FID / FVD
- **跨语言泛化**: 非英语视频的 lip-sync vs 英语视频的 lip-sync 差距

### 训练规模

每组实验：
- 数据: 37-45 个视频
- GPU: 1× RTX PRO 6000 或多卡
- 迭代: 至少 1000-3000 iterations（100 iter 只够验证 pipeline，不够出效果）
- 预计时间: 1000 iter ≈ 2.5h (单卡), 3000 iter ≈ 7.5h

### 前置工作

1. **实验 A (Baseline)**: 用原始 wav2vec 重新提取 audio_emb `(T, 12, 768)`，用原始 AudioProjModel 参数训练
2. **实验 B**: 已就绪（当前代码 + LTX-2 VAE embeddings）
3. **实验 C/D**: 需要收集多语言训练数据
4. **推理脚本适配**: `sample_video.py` 的 AudioProcessor 需要改成 LTX-2 VAE 版本（当前仍用 wav2vec）
5. **SyncNet 评估**: 需要部署评估工具到新集群

### 建议优先级

```
Phase 1: B vs A （只换编码器，同数据）
  → 验证编码器替换的独立效果
  → 需要: 跑 A baseline + B 完整训练 + 推理 + 评估

Phase 2: C vs A （只换数据，同编码器）
  → 验证多语言数据的独立效果
  → 需要: 收集多语言数据

Phase 3: D vs A/B/C
  → 验证两者叠加效果
```

---

## 代码修改总览

```
修改的文件:
  hallo3/sgm/models/transformer.py      ← AudioProjModel 维度 (blocks=1, channels=128)
  hallo3/data_video.py                  ← unsqueeze + weights_only=False
  hallo3/sample_video.py                ← unsqueeze
  hallo3/train_video.py                 ← checkpoint 加载 shape mismatch 兼容

新增的文件:
  hallo3/sgm/models/ltx_audio_vae/      ← LTX-2 VAE 模块 (11 文件)
  scripts/extract_audio_emb_ltx_vae.py  ← 音频 embedding 提取
  scripts/verify_ltx_audio_vae.py       ← VAE 独立验证
  scripts/preprocess_and_extract.sh     ← 预处理 SLURM 脚本
  scripts/verify_train_100iter.sh       ← 训练验证 SLURM 脚本
  configs/sft_s2_verify.yaml            ← 100 iter 验证配置

未修改:
  hallo3/sgm/models/wav2vec.py          ← 保留旧编码器用于消融对比
  hallo3/sgm/utils/audio_processor.py   ← 未修改（推理时仍用旧 AudioProcessor）
  hallo3/dit_video_concat.py            ← Cross-Attention 层不需要改
```

---

## 新集群信息

- 集群：AICR (`login.aicr.ai`)
- 账户：`p2026_0014_neu`
- GPU：RTX PRO 6000 Blackwell (sm_120)
- PyTorch：2.7.1+cu128（Blackwell 需要 cu128）
- Conda 环境：`hallo3` (Python 3.10)
- DeepSpeed: 需用 `DS_BUILD_OPS=0` 安装（Blackwell 不支持 FusedAdam 编译）
- Optimizer: 用标准 Adam 替代 FusedEmaAdam
- 详细环境搭建问题见 `cluster_setup_log.md`
