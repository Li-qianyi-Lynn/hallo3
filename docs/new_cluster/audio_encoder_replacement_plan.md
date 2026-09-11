# Audio Encoder Replacement Plan: Wav2Vec2 → LTX-2 Audio VAE

## 动机

Hallo3 现有的 `wav2vec2-base-960h` 在 960 小时**纯英语** LibriSpeech 上预训练，学到的是英语音素表征。处理非英语语音时特征质量下降，导致 lip-sync 在非英语上表现差。

LTX-2 的 Audio VAE 工作在 **mel spectrogram** 上（纯声学信号：频率 + 能量 + 时间），与语言无关，天然具备跨语言泛化能力。

```
Wav2Vec2:   waveform → 英语音素特征 (language-biased)
LTX-2 VAE:  waveform → mel → 声学潜在表示 (language-agnostic)
```

---

## 两个编码器的对比

| 维度 | Hallo3 (Wav2Vec2) | LTX-2 (Audio VAE) |
|------|-------------------|-------------------|
| 模型类型 | 预训练 Transformer (12 层) | 卷积 VAE (ResNet + Self-Attention) |
| 特征类型 | 语音语义特征（英语音素） | 压缩声学潜在表示（语言无关） |
| 采样率 | 16kHz | 16kHz (一致) |
| 时间分辨率 | 25fps（线性插值） | **25fps（自然对齐：16000/160/4=25）** |
| 输出 blocks | 12 层 hidden states | 1 层 latent（无多层结构） |
| 输出维度/帧 | 12 × 768 = 9,216 值 | 8 ch × 16 mel_bins = **128 值** |
| 信息量比 | 基准 | **72× 压缩** |
| 预训练数据 | 英语 LibriSpeech 960h | 语言无关 (mel spectrogram) |

---

## LTX-2 AudioEncoder 内部架构（源码验证）

### 下采样路径（关键！旧版计划遗漏了空间下采样）

```
配置: ch=128, ch_mult=(1, 2, 4), z_channels=8, double_z=True
      in_channels=2 (stereo), mel_bins=64
      norm_type=PIXEL, causality_axis=HEIGHT

输入 mel spectrogram: (B, 2, T_mel, 64)
    ↓ conv_in: CausalConv2d(2→128, kernel=3)
    ↓
Level 0: 2× ResnetBlock(128→128)
    ↓ Downsample(stride=2)                    ← 时间 ÷2, 频率 ÷2
    ↓ (B, 128, T_mel/2, 32)

Level 1: 2× ResnetBlock(128→256)
    ↓ Downsample(stride=2)                    ← 时间 ÷2, 频率 ÷2
    ↓ (B, 256, T_mel/4, 16)

Level 2: 2× ResnetBlock(256→512)
    ↓ NO Downsample（最后一层不下采样）
    ↓ (B, 512, T_mel/4, 16)

Mid block: ResnetBlock(512) → AttnBlock(512) → ResnetBlock(512)
    ↓ (B, 512, T_mel/4, 16)

norm_out + SiLU + conv_out(512→16, kernel=3)   [double_z: 2×8=16]
    ↓ (B, 16, T_mel/4, 16)

取 mean (前半 channels):
    ↓ (B, 8, T_mel/4, 16)

Patchify: rearrange "b c t f -> b t (c f)"
    ↓ (B, T_mel/4, 128)           ← 128 = 8 × 16

PerChannelStatistics.normalize():
    ↓ (B, T_mel/4, 128)           ← 最终输出
```

**总下采样倍数: 4× (时间和频率都是)**

代码依据:
- `audio_vae.py:213`: `if level != self.num_resolutions - 1: h = stage.downsample(h)`
- `downsample.py:34`: `Conv2d(in_channels, in_channels, kernel_size=3, stride=2)`
- `patchifiers.py:301`: `rearrange(audio_latents, "b c t f -> b t (c f)")`
- `audio_vae.py:19`: `LATENT_DOWNSAMPLE_FACTOR = 4`

### 时间分辨率计算

```
Mel 帧率 = sample_rate / hop_length = 16000 / 160 = 100 fps
Latent 帧率 = Mel 帧率 / downsample_factor = 100 / 4 = 25 fps  ← 与视频帧率天然对齐！
```

以 10 秒音频为例:
- Wav samples: 160,000
- Mel frames: 160,000 / 160 = 1,000
- Latent frames: 1,000 / 4 = **250 帧 (= 10s × 25fps)**

---

## 适配路径（修正版）

```
LTX-2 AudioEncoder                          Hallo3 现有管线
─────────────────                          ──────────────────
waveform (16kHz)                           waveform (16kHz)
  ↓                                          ↓
mel spectrogram (B, 2, T_mel, 64)          Wav2Vec2 (B, T, 12, 768)     ← 替换掉
  ↓                                          ↓
AudioEncoder VAE                           时间窗口 ±2
  ↓ 4× 下采样（时间+频率）                    ↓ (T, 5, 12, 768)
  ↓ (B, 8, T_mel/4, 16)                     ↓
  ↓                                        AudioProjModel MLP
patchify + normalize                         input: 5×12×768 = 46,080
  ↓ (B, T_mel/4, 128)                       ↓
  ↓                                        → 32 × 768 context tokens
unsqueeze blocks dim                         ↓
  ↓ (B, T, 1, 128)                        Cross-Attention in DiT
  ↓
时间窗口 ±2
  ↓ (frames, 5, 1, 128)
  ↓
AudioProjModel MLP (修改后)
  input: 5×1×128 = 640                      ← 原来 46,080
  ↓
→ 32 × 768 context tokens                   ← 保持不变
  ↓
Cross-Attention in DiT                       ← 保持不变
```

---

## Step 1: 下载 LTX-2 音频 VAE 权重

```bash
# 在集群上
pip install huggingface_hub
huggingface-cli download Lightricks/LTX-2.5 \
    vae/ltx-2.5-audio-vae-bf16.safetensors \
    --local-dir /scratch/li_qiany_neu/pretrained_models/ltx2.5
```

- HuggingFace 仓库: `Lightricks/LTX-2.5`
- 权重文件: `vae/ltx-2.5-audio-vae-bf16.safetensors`
- 格式: safetensors, 包含 encoder/decoder state_dict + PerChannelStatistics

---

## Step 2: 将 LTX-2 audio_vae 模块搬进 Hallo3

### 需要复制的文件

源目录: `LTX-2/packages/ltx-core/src/ltx_core/`

| 文件 | 来源 | 作用 |
|------|------|------|
| `audio_vae.py` | `model/audio_vae/audio_vae.py` | AudioEncoder 主体（只需 encoder，不需 decoder） |
| `ops.py` | `model/audio_vae/ops.py` | AudioProcessor (waveform→mel) + PerChannelStatistics |
| `resnet.py` | `model/audio_vae/resnet.py` | ResnetBlock 残差块 |
| `causal_conv_2d.py` | `model/audio_vae/causal_conv_2d.py` | 因果 2D 卷积 |
| `causality_axis.py` | `model/audio_vae/causality_axis.py` | CausalityAxis 枚举 |
| `attention.py` | `model/audio_vae/attention.py` | Self-Attention (AttnBlock) |
| `downsample.py` | `model/audio_vae/downsample.py` | Downsample + build_downsampling_path |
| `normalization.py` | `model/common/normalization.py` | PixelNorm, NormType, build_normalization_layer |
| `patchifier.py` | `components/patchifiers.py` | AudioPatchifier (patchify/unpatchify) |
| `types.py` | `types.py` | AudioLatentShape, Audio dataclass |

### 不需要复制的文件

| 文件 | 原因 |
|------|------|
| `vocoder.py` | 只需 encoder，不做 audio 重建 |
| `upsample.py` | decoder 专用 |
| `model_configurator.py` | 可简化为直接实例化，不需要完整配置系统 |

### 建议放置位置

```
hallo3/sgm/models/
├── wav2vec.py              # 原有，保留用于对比实验
└── ltx_audio_vae/          # 新增
    ├── __init__.py
    ├── audio_vae.py        # AudioEncoder (删除 AudioDecoder 相关代码)
    ├── ops.py              # AudioProcessor + PerChannelStatistics
    ├── resnet.py
    ├── causal_conv_2d.py
    ├── causality_axis.py
    ├── attention.py
    ├── downsample.py
    ├── normalization.py
    ├── patchifier.py       # AudioPatchifier
    └── types.py            # AudioLatentShape, Audio
```

### Import 路径修改

所有文件中的 `from ltx_core.xxx` 需改为相对导入 `from .xxx`，例如:
```python
# 旧: from ltx_core.model.audio_vae.resnet import ResnetBlock
# 新: from .resnet import ResnetBlock

# 旧: from ltx_core.model.common.normalization import NormType
# 新: from .normalization import NormType

# 旧: from ltx_core.components.patchifiers import AudioPatchifier
# 新: from .patchifier import AudioPatchifier

# 旧: from ltx_core.types import AudioLatentShape, Audio
# 新: from .types import AudioLatentShape, Audio
```

### 额外依赖

需要确保集群环境有 `torchaudio`（LTX-2 的 mel 提取用 `torchaudio.transforms.MelSpectrogram`）:
```bash
pip install torchaudio
```

---

## Step 3: 修改 Hallo3 的三个关键文件

### 3.1 `hallo3/sgm/utils/audio_processor.py`

替换预处理逻辑:

```python
# 旧流程:
#   waveform → Wav2Vec2FeatureExtractor → Wav2Vec2Model → (T, 12, 768)
#
# 新流程:
#   waveform → AudioProcessor.waveform_to_mel → AudioEncoder.forward()
#   → (B, 8, T_mel/4, 16) [normalize 已内含]
#   → 手动 patchify: rearrange "b c t f -> b t (c f)"
#   → (T, 128)
#
#   注意: T = T_mel/4 = 25fps，自然对齐视频帧率，无需插值
```

关键变化:
- 移除 `Wav2VecModel.from_pretrained()` 和 `Wav2Vec2FeatureExtractor`
- 新增 `AudioProcessor` (mel 转换, from `ops.py`)
- 新增 `AudioEncoder` (VAE 编码, from `audio_vae.py`)
- 输出从 `(time, 12, 768)` 变为 **`(time, 128)`**
- 不再需要 `linear_interpolation` 到 fps（VAE 自然输出 25fps）

### 3.2 `hallo3/sgm/models/transformer.py` — AudioProjModel (line 345)

修改输入维度:

```python
class AudioProjModel(torch.nn.Module):
    def __init__(
        self,
        seq_len=5,
        blocks=1,            # 原来 12 → 改为 1（VAE 没有多层 hidden states）
        channels=128,        # 原来 768 → 改为 128（VAE patchified dim）
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32
    ):
        # input_dim = 5 × 1 × 128 = 640  (原来 5 × 12 × 768 = 46,080)
        self.input_dim = seq_len * blocks * channels  # 640

        self.proj1 = Linear(640, 512)    # 原来 Linear(46080, 512)
        self.proj2 = Linear(512, 512)    # 不变
        self.proj3 = Linear(512, 32 * 768)  # 不变，输出仍然是 32 × 768
```

forward 中的 reshape 逻辑不变: `(bz f) w b c -> flatten -> proj -> 32 × 768`

### 3.3 `hallo3/data_video.py` — 训练数据加载 (line 715-745)

修改 audio embedding 的加载和 temporal window 构建:

```python
# 旧: audio_emb shape = (T, 12, 768)
#     window → audio_emb[center_indices] → (num_frames, 5, 12, 768)

# 新: audio_emb shape = (T, 128)
#     需要 unsqueeze blocks 维度:
#     audio_emb = audio_emb.unsqueeze(1)  → (T, 1, 128)
#     window → audio_emb[center_indices] → (num_frames, 5, 1, 128)
#     这样 AudioProjModel 的 forward 中 "bz f w b c" 仍然兼容
```

---

## Step 4: 重新提取 audio embeddings

用新编码器对训练数据重新跑:

```bash
# 重新提取 42 个 pilot 视频的 audio embeddings
python hallo3/data_preprocess.py \
    --input_dir /scratch/li.qianyi/hallo3_data/talkvid/ \
    --audio_encoder ltx_vae \
    --audio_vae_path /scratch/li.qianyi/pretrained_models/ltx2.5/vae/ltx-2.5-audio-vae-bf16.safetensors
```

新的 `.pt` 文件 shape: **`(T, 128)`** (原来是 `(T, 12, 768)`)

以 10 秒视频为例:
- 旧: `(250, 12, 768)` = 2,304,000 值, ~8.8 MB (fp32)
- 新: `(250, 128)` = 32,000 值, ~0.12 MB (fp32)
- 存储节省: **72×**

---

## Step 5: 小规模验证（分两阶段）

### 5.1 独立验证（先跑通 encoder）

在本地或集群上写一个简单脚本，验证 AudioEncoder 能独立运行:

```python
import torch
from sgm.models.ltx_audio_vae import AudioEncoder, AudioProcessor

# 加载权重
encoder = load_audio_encoder("path/to/ltx-2.5-audio-vae-bf16.safetensors")
processor = AudioProcessor(target_sample_rate=16000, mel_bins=64, mel_hop_length=160, n_fft=1024)

# 10 秒音频
waveform = torch.randn(1, 2, 160000)  # (batch, stereo, samples)
mel = processor.waveform_to_mel(Audio(waveform=waveform, sampling_rate=16000))
print(f"Mel shape: {mel.shape}")       # 期望: (1, 2, 1000, 64)

latent = encoder(mel)
print(f"Latent shape: {latent.shape}") # 期望: (1, 8, 250, 16)

# 手动 patchify
from einops import rearrange
tokens = rearrange(latent, "b c t f -> b t (c f)")
print(f"Token shape: {tokens.shape}")  # 期望: (1, 250, 128)
```

### 5.2 端到端验证

1. 100 iterations, 42 videos, 1× GPU
2. 确认 forward pass + backward pass 无报错
3. 跑一次 inference，确认能生成视频（质量先不管）
4. 对比 loss 曲线是否正常下降

---

## 验证完成后 → 进入消融实验

| # | 编码器 | 数据 | 说明 |
|---|--------|------|------|
| A | Wav2Vec2 (原) | 英语为主 | Baseline |
| B | LTX-2 VAE (新) | 英语为主 | 隔离编码器贡献 |
| C | Wav2Vec2 (原) | 平衡多语言 | 隔离数据贡献 |
| D | LTX-2 VAE (新) | 平衡多语言 | 完整方案 |

---

## 风险评估

| 风险 | 级别 | 详细说明 | 缓解措施 |
|------|------|----------|----------|
| **信息瓶颈** | 中 | 每帧特征从 9,216 值降到 128 值 (72×)。上游信息量大幅减少 | 若效果不好，可尝试: (1) 增大 intermediate_dim 到 1024; (2) 去掉 patchify 直接用 (8, 16) 2D 特征; (3) 加更多 MLP 层 |
| **多层语义丢失** | 中 | Wav2Vec2 的 12 层提供了从底层声学到高层语义的梯度信息。VAE 只有单层 latent | 这是 trade-off: 丢失多粒度语义，换来语言无关性。消融实验会量化这个 trade-off |
| **AudioProjModel 行为变化** | 低 | proj1 从 46k→512 (压缩 90×) 变成 640→512 (压缩 1.25×)，网络动态不同 | 初始 lr 可能需要调整。proj1 不再是信息压缩层，而是接近恒等映射 |
| **依赖文件搬运** | 低 | ~10 个文件的 import 路径需要修改 | 约 1-2 小时工作量，已列出完整清单 |
| **torchaudio 依赖** | 低 | LTX-2 用 `torchaudio.transforms.MelSpectrogram` | 确保集群 `pip install torchaudio` |
| **Stereo vs Mono** | 低 | LTX-2 AudioEncoder 输入是 stereo (2 channels)，Hallo3 原来用 mono | librosa 加载时用 `mono=False`，或将 mono 复制成 stereo: `waveform.repeat(1, 2, 1)` |

---

## 完整数据流

```
Raw audio (16kHz, mono→stereo)
  ↓ torchaudio resample (if needed)
  ↓ MelSpectrogram(n_fft=1024, hop=160, n_mels=64)
  ↓ log(clamp(mel, 1e-5)) → permute
  ↓ → (B, 2, T_mel, 64)                               [T_mel=100fps]
  ↓
AudioEncoder:
  ↓ conv_in(2→128) → 3 levels (ch_mult=1,2,4)
  ↓ 2× Downsample(stride=2): 时间÷4, 频率÷4
  ↓ mid block (ResNet+Attention)
  ↓ conv_out(512→16) → take mean → (B, 8, T_mel/4, 16) [T=25fps]
  ↓ patchify + PerChannelStatistics normalize
  ↓ → (T, 128)                                         [T=25fps]
  ↓
保存 .pt → DataLoader 加载
  ↓ unsqueeze blocks: (T, 1, 128)
  ↓ temporal window ±2 → (num_frames, 5, 1, 128)
  ↓
AudioProjModel (修改后):
  ↓ flatten: 5×1×128 = 640
  ↓ proj1: Linear(640→512) + ReLU                      ← 改这里
  ↓ proj2: Linear(512→512) + ReLU                      ← 不变
  ↓ proj3: Linear(512→32×768) → reshape (32, 768)      ← 不变
  ↓ Conv1D pooling ×2 + LayerNorm                      ← 不变
  ↓ → (B, frames, 32, 768)                             ← 不变
  ↓
DiT 每层 Cross-Attention: Query=video (1024d), KV=audio (32×768)  ← 不变
```
