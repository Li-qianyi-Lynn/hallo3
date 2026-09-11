# Audio Encoder 替换进度

日期：2026-09-10

---

## 总体进度

| Step | 任务 | 状态 |
|------|------|------|
| 1 | 下载 LTX-2 音频 VAE 权重 | 完成 |
| 2 | 将 LTX-2 audio_vae 模块搬进 Hallo3 | 完成 |
| 3 | 修改 Hallo3 pipeline 文件 | 完成 |
| 4 | 重新提取 audio embeddings | 排队中 |
| 5 | 小规模验证 (100 iterations) | 未开始 |

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

### 3.1 `hallo3/sgm/models/transformer.py` — AudioProjModel

```python
# 旧默认值:
blocks=12, channels=768  → input_dim = 5×12×768 = 46,080

# 新默认值:
blocks=1, channels=128   → input_dim = 5×1×128 = 640
```

### 3.2 `hallo3/data_video.py` — 训练数据加载（两处）

添加自动兼容逻辑：
```python
# LTX-2 VAE: (frames, 5, 128) → unsqueeze → (frames, 5, 1, 128)
if audio_tensor.dim() == 3:
    audio_tensor = audio_tensor.unsqueeze(2)
```

### 3.3 `hallo3/sample_video.py` — 推理 process_audio_emb

同样添加 unsqueeze 逻辑，兼容新旧格式。

### 3.4 新增 `scripts/extract_audio_emb_ltx_vae.py`

用 LTX-2 VAE 从音频文件提取 embeddings，输出 `(T, 128)` 的 .pt 文件。

## Step 4: 重新提取 audio embeddings — 排队中

### 数据迁移
- 45 个 pilot 视频 (.mp4) 从旧集群传到新集群
- 位置：`/scratch/li_qiany_neu/TalkVidTestData/clips_flat/`
- Hallo3 预训练权重已下载到 `/scratch/li_qiany_neu/pretrained_models/`

### 提交的 SLURM job
- Job ID: 793239
- 分区: rtx-devel
- 状态: PD (Pending，等待节点空闲)
- 脚本: `scripts/preprocess_and_extract.sh`

脚本流程：
1. `data_preprocess.py` — 从 mp4 提取 face_emb、face_mask、audio (.wav)
2. `extract_audio_emb_ltx_vae.py` — 用 LTX-2 VAE 从 .wav 提取 audio_emb `(T, 128)`

预期输出目录结构：
```
/scratch/li_qiany_neu/TalkVidTestData/
├── clips_flat/          # 45 个 mp4 视频
├── audios/              # 从 mp4 提取的 .wav
├── face_emb/            # 人脸特征 .pt
├── face_mask/           # 人脸 mask .png
└── audio_emb/           # LTX-2 VAE audio embeddings .pt (T, 128)
```

## Step 5: 小规模验证 — 未开始

等 Step 4 完成后：
1. 100 iterations, 45 videos, 1× GPU
2. 确认 forward + backward pass 无报错
3. 确认 loss 正常下降
4. 跑一次 inference 生成视频

---

## 新集群信息

- 集群：AICR (`login.aicr.ai`)
- 账户：`p2026_0014_neu`
- GPU：RTX PRO 6000 Blackwell (sm_120)
- PyTorch：2.7.1+cu128（Blackwell 需要 cu128）
- Conda 环境：`hallo3` (Python 3.10)
- 详细环境搭建问题见 `cluster_setup_log.md`
