# 2026-10-16：新 encoder 效果差的根因分析与改进方案

承接 [2026-10-09_syncnet_val5_gt_vs_gen.md](../progress/2026-10-09_syncnet_val5_gt_vs_gen.md)。

## 一、问题回顾

新 encoder（LTX-2 VAE）替换原版 wav2vec 后，SyncNet 打分严重退化：

| | 真值 | 原版 Hallo3 | 新 encoder 7000 步 |
|---|------|-------------|---------------------|
| Min dist 平均 | 7.55 | 9.29 | 13.65 |
| Confidence 平均 | 7.12 | 6.33 | 1.04 |
| offset | 稳定 -1...-2 | 稳定 -1...-2 | 乱跳 -15...+15 |

新 encoder 的 conf 只有原版的 ~16%，口型基本没被音频驱动。

---

## 二、根因分析

### 2.1 信息瓶颈：输入维度骤降 73 倍

AudioProjModel 的输入维度变化：

| | 原版 (wav2vec) | 新 encoder (LTX-2 VAE) |
|---|---|---|
| 每帧特征维度 | 768 | 128 |
| 堆叠层数 (blocks) | 12 | 1 |
| 窗口大小 (seq_len) | 5 | 5 |
| **proj1 输入维度** | **5 x 12 x 768 = 46,080** | **5 x 1 x 128 = 640** |
| proj1 输出维度 | 512 | 512 |
| 最终输出 | 32 tokens x 768 dim | 32 tokens x 768 dim |

proj1 原来是 46,080 -> 512 的**压缩层**（从大量冗余中提取精华），现在变成 640 -> 512 的**几乎无压缩传递**，但后面 proj3 还是要从 512 -> 32x768 = 24,576 展开。信息不够，网络只能学到近似常量的输出。

### 2.2 表征目标不匹配

| | wav2vec2 | LTX-2 Audio VAE |
|---|---|---|
| 训练目标 | 自监督语音表征学习（对比学习 + masked prediction） | 音频重建（VAE：encode -> decode -> 重建 mel 谱） |
| 特征含义 | 音素、韵律、语义（12 层金字塔） | 声学纹理/音色（压缩码，用于重建） |
| 层数 | 12 层，每层 768 维 | 1 层 latent，8 channels x 16 mel = 128 维 |
| 设计目的 | 区分语音内容 | 压缩和重建音频信号 |

wav2vec 的 12 层堆叠是 Hallo 系列的刻意设计（见 [Hallo2 论文](https://arxiv.org/html/2410.07718)）：
- 底层（1-4）：声学/音素特征 -> 嘴巴张合的物理信号
- 中层（5-8）：音节/韵律特征 -> 节奏、重音
- 高层（9-12）：语义特征 -> 词义上下文

LTX-2 VAE 的 latent 是为了**重建 mel 谱**设计的，不保证保留口型驱动所需的音素级时序信息。

### 2.3 预训练权重丢失

`train_video.py` 的 monkey-patch 逻辑在加载 checkpoint 时，因为 shape 不匹配跳过了 `audio_proj.proj1.weight`（512x46080 vs 512x640）。AudioProjModel 是**从随机初始化开始训的**，而 DiT backbone 和 cross-attention 层都保留了预训练权重。

7000 步 + 2702 条数据对于随机初始化的投影层来说训练量严重不足。

### 2.4 LTX-2 内部的 audio 使用方式与 Hallo3 完全不同

在 LTX-2 自身的架构里（[LTX-2 论文](https://arxiv.org/pdf/2601.03233)）：
- audio latent 作为**平行 token 序列**和 video latent 一起处理
- 每一层都做 **双向 cross-attention（audio <-> video）**
- audio latent 直接保持 (B, T, C) 的序列形式，不做 flatten

而 Hallo3 的用法是：把 audio latent **flatten 成一个大向量** -> 3 层 MLP -> 生成 32 个 context tokens -> 单向 cross-attention（video query, audio key/value）。这完全不是 LTX-2 audio VAE 的设计用途。

---

## 三、文献支撑

### 3.1 音频编码器的预训练目标决定适配性

[Audio-driven Talking Face Generation with Stabilized Synchronization Loss (ECCV 2024)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02950.pdf)：

- 作者使用与人脸编码器**联合预训练**（针对音画同步目标）并完全冻结的 AVSyncNet 音频编码器，直接提取 512 维音素级特征，无需额外的特征空间对齐模块
- 消融实验（Section 4.3, Table 2, Setup C vs B）中替换为 Wav2Vec 2.0 后，各项指标仅**轻微下降**（PSNR 26.614->26.590, SSIM 0.868->0.860, LMD 2.325->2.278），并非严重崩溃
- 作者的归因：音频编码器最好是**专门为音视频同步目标预训练**的，因此 AVSyncNet 训练好的音频分支比通用语音特征提取器（wav2vec）更适配

**对我们的启示**：该论文的发现是"为 lip sync 目标预训练的 encoder > 通用语音 encoder"，而我们的情况更极端——LTX-2 VAE 是为**音频重建**目标训练的，连通用语音表征都不是，离 lip sync 目标更远。wav2vec 至少还保留了音素级时序信息，VAE latent 的设计目标是压缩重建，不保证保留口型驱动所需的区分性特征。

### 3.2 多层特征聚合的重要性

[Dual Audio-Centric Modality Coupling for Talking Head Generation](https://arxiv.org/html/2503.22728v1)：multi-layer wav2vec features are aggregated via projection layers to fuse audio cues from different semantic levels。单层 latent 丢失了这种多层次互补信息。

### 3.3 低维特征需要 expansion

[Multimodal Feature-Guided Emotional Talking Face Generation (2025)](https://www.mdpi.com/2079-9292/14/13/2684)：encoder maps low-dimensional AU representations into high-dimensional continuous embeddings while preserving feature granularity through residual learning。低维到高维的扩展需要专门设计。

---

## 四、改进方案

### 方案 A：加 Expansion MLP（最小改动，优先尝试）

在 AudioProjModel 的 flatten 后、proj1 前加特征扩展层：

```
现在（有问题）:
  LTX VAE -> (T, 128) -> 窗口采样 -> flatten (640) -> proj1(640->512) -> proj2 -> proj3 -> 32x768

改后:
  LTX VAE -> (T, 128) -> 窗口采样 -> flatten (640)
    -> expansion(640->2048) -> GELU -> expansion2(2048->2048) -> GELU
    -> proj1(2048->512) -> ReLU -> proj2 -> proj3 -> 32x768
```

代码改动（`hallo3/sgm/models/transformer.py`，AudioProjModel 类）：

```python
# __init__ 里加：
self.expansion = torch.nn.Sequential(
    torch.nn.Linear(self.input_dim, 2048),   # 640 -> 2048
    torch.nn.GELU(),
    torch.nn.Linear(2048, 2048),              # 2048 -> 2048
    torch.nn.GELU(),
)
# proj1 改成：
self.proj1 = torch.nn.Linear(2048, intermediate_dim)  # 2048 -> 512

# forward 里，在 flatten 之后、proj1 之前加：
audio_embeds = self.expansion(audio_embeds)  # (B, 640) -> (B, 2048)
```

新增参数：~540 万（640x2048 + 2048 + 2048x2048 + 2048 + 2048x512 + 512）

### 方案 B：加 Transformer 层做 per-frame 特征增强（更强但更贵）

在窗口采样之前，对 LTX VAE 输出的 (T, 128) 序列做 self-attention + 维度扩展：

```python
class AudioFeatureEnhancer(torch.nn.Module):
    def __init__(self, in_dim=128, out_dim=768, num_layers=2, nhead=8):
        super().__init__()
        self.proj_in = torch.nn.Linear(in_dim, out_dim)  # 128 -> 768
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=out_dim, nhead=nhead, dim_feedforward=out_dim*4,
            batch_first=True, activation='gelu'
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, x):
        # x: (B, T, 128)
        x = self.proj_in(x)     # -> (B, T, 768)
        x = self.transformer(x) # -> (B, T, 768) with temporal context
        return x
```

这样 AudioProjModel 的输入变成 blocks=1, channels=768，input_dim = 5x1x768 = 3840。每帧有 768 维表征 + 跨帧时序上下文。新增参数 ~1900 万。

### 方案对比

| | 方案 A (Expansion MLP) | 方案 B (Transformer) |
|---|---|---|
| 改动量 | 小，只改 AudioProjModel | 中等，加新模块 + 改数据流 |
| 新增参数 | ~540 万 | ~1900 万 |
| 能力 | 扩展维度，无跨帧上下文 | 维度 + 跨帧时序上下文 |
| 风险 | 低 | 需调学习率，可能需更多训练步数 |
| 建议 | **先试这个** | 方案 A 有效后再升级 |

---

## 五、实验计划

1. **方案 A 实验**：修改 AudioProjModel，加 expansion 层，同样 2702 条数据训 7000 步，同样 5 条 val 打 SyncNet
2. **对照组**（research plan 里的 A 组）：原 wav2vec encoder，同样 2702 条，同样步数，排除训练量因素
3. 如果方案 A 的 conf 显著提升（目标 > 3），再尝试方案 B
4. 如果两个方案都不行，考虑放弃 LTX-2 VAE，换用多语言语音编码器（如 Whisper encoder，天然支持多语言 + 有语义层次）

---

## 六、关键代码位置

| 文件 | 行号 | 内容 |
|------|------|------|
| `hallo3/sgm/models/transformer.py` | 348-433 | AudioProjModel 定义（需修改） |
| `hallo3/sgm/utils/ltx_audio_processor.py` | 1-105 | LTX-2 VAE 音频处理 |
| `hallo3/sgm/models/ltx_audio_vae/audio_vae.py` | 56-224 | AudioEncoder VAE 实现 |
| `hallo3/dit_video_concat.py` | 576-587 | Cross-attention 注入点 |
| `hallo3/train_video.py` | 8-67 | Checkpoint 加载 monkey-patch |
