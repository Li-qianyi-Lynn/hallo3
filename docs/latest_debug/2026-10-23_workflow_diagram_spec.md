# 2026-10-23：Hallo3 + Omnilingual Encoder Workflow 画图规格

本文档列出画 workflow 图需要涉及的所有细节，包括每一步的模块名、输入输出 shape、关键参数和新旧对比标注。

---

## 一、完整 workflow（改版后）

图应包含以下 **8 个阶段**，从上到下或从左到右：

### Stage 1: Audio Input

```
输入: 音频波形 (.wav), 16kHz, mono
shape: (1, N_samples)
例如 5 秒音频: (1, 80000)
```

### Stage 2: Omnilingual wav2vec2 Encoder（frozen, 不参与训练）

```
模块: Wav2VecModel.from_pretrained("facebook/omnilingual-w2v2-300m")
      ├── CNN Feature Encoder: 波形 → 512 维, 7 层 1D CNN
      ├── Linear Projection: 512 → 1024
      └── Transformer Encoder: 24 层, 每层 1024 维, Self-Attention

输入: (1, 80000) raw waveform
输出: hidden_states — 25 个 tensor (embedding + 24 layers)
      取最后 12 层: (T, 12, 1024)
      T = ceil(duration * fps), 例如 5s × 25fps = 125 帧

标注:
  - 🔒 frozen (不参与 gradient 计算)
  - 🆕 替换原版 wav2vec2-base (768 dim, 12 layers)
  - training objective: contrastive learning (InfoNCE loss)
  - 取最后 12/24 层 (代码: audio_processor.py, hidden_states[-12:])
```

### Stage 3: Sliding Window Sampling

```
操作: 对每个视频帧 i, 取音频帧 [i-2, i-1, i, i+1, i+2]
      窗口大小 = 5 (当前帧 ± 2 帧)

输入: (T, 12, 1024)
输出: (B, F, 5, 12, 1024)
      B = batch size, F = 视频帧数

标注:
  - 纯数据操作，无可学习参数
  - 目的: 给每帧提供时序上下文
```

### Stage 4: Adapter Layer（🆕 新增，随机初始化，参与训练）

```
模块: nn.Linear(1024, 768)
      参数量: 1024 × 768 + 768 ≈ 790K

输入: (B, F, 5, 12, 1024)
输出: (B, F, 5, 12, 768)

标注:
  - 🆕 新增层 (原版没有)
  - 🔥 参与训练 (随机初始化, ~79 万参数)
  - 作用: dimensionality adaptation, 把 Omnilingual 的 1024 维
          映射到原版 wav2vec2-base 的 768 维空间
  - 位于 AudioProjModel 内部, flatten 之前
  - checkpoint 加载时: missing key, 自动随机初始化
```

### Stage 5: Flatten + AudioProjModel MLP（加载 pretrained weights）

```
子步骤 5a — Flatten:
  操作: rearrange + view
  输入: (B*F, 5, 12, 768)
  输出: (B*F, 46080)     ← 5 × 12 × 768 = 46,080

子步骤 5b — proj1:
  模块: nn.Linear(46080, 512) + ReLU
  输入: (B*F, 46080)
  输出: (B*F, 512)
  作用: 压缩 — 从 46,080 维中 extract 512 维精华
       512 个 "探测器", 每个负责检测不同的语音特征
       (phoneme type, prosody, temporal dynamics, etc.)

子步骤 5c — proj2:
  模块: nn.Linear(512, 512) + ReLU
  输入: (B*F, 512)
  输出: (B*F, 512)
  作用: 非线性变换, 增强表达能力 (Universal Approximation Theorem)

子步骤 5d — proj3:
  模块: nn.Linear(512, 32×768) + reshape
  输入: (B*F, 512)
  输出: (B*F, 32, 768)
  作用: 展开成 32 个 context tokens
       每个 token 768 维, 和 DiT hidden dim 对齐

标注:
  - ✅ proj1/proj2/proj3 全部加载 pretrained weights (shape 完全匹配)
  - 🔥 参与训练 (fine-tune)
  - proj1.weight shape: (512, 46080) — 和原版 checkpoint 完全一致, 零 mismatch
```

### Stage 6: Conv1d Temporal Downsampling

```
模块: nn.Conv1d(24576, 24576, kernel=2, stride=2), 执行 2 次

输入: (B, F, 32×768) = (B, F, 24576)
输出: (B, F', 32, 768)
      F' ≈ F/4 (两次减半)

例如: F=13 → 7 → 4

标注:
  - ✅ 加载 pretrained weights
  - 作用: 将音频帧率对齐到 DiT 的 latent frame rate
  - + LayerNorm 归一化
```

### Stage 7: Cross-Attention Injection into DiT

```
在 DiT 的每一层 transformer block 中:

  Q = img_hidden_states (视频特征)
      shape: (B*T, N_patches, 768)

  K, V = audio context tokens
      shape: (B*T, 32, 768)

  操作:
    audio_attn_input = LayerNorm(img_hidden_states)
    audio_attn_output = CrossAttention(Q=视频, K=音频, V=音频)
    img_hidden_states = img_hidden_states + audio_attn_output  ← residual connection

标注:
  - ✅ 加载 pretrained weights
  - 每一层 DiT 都做一次 (多尺度注入)
  - 低层: 控制细节 (嘴唇纹理)
  - 高层: 控制结构 (嘴巴形状)
```

### Stage 8: DiT Diffusion Output

```
DiT backbone (frozen / partially fine-tuned):
  输入: noisy latent + text prompt + audio conditioning + face reference
  输出: denoised video latent

Video VAE Decoder:
  输入: denoised latent
  输出: 生成的视频帧 (带口型)
```

---

## 二、图中需要标注的对比信息

### 新旧对比标记

在图的右侧或旁边标注新旧对比：

| 位置 | 原版 Hallo3 | 改版后 |
|---|---|---|
| Stage 2 Encoder | wav2vec2-base (768d, 12L, 95M params) | Omnilingual wav2vec2 (1024d, 24L, 300M params) |
| Stage 2 取层 | 全部 12 层 | 最后 12/24 层 |
| Stage 4 Adapter | 不存在 | 🆕 Linear(1024→768), ~790K params |
| Stage 5a Flatten | 5×12×768 = 46,080 | 5×12×768 = 46,080 (adapter 之后, 一致) |
| Stage 5b proj1 | pretrained, (512, 46080) | ✅ 同样 pretrained, 零 mismatch |

### 之前 LTX-2 VAE 方案（可选，标注为 ✗ 已放弃）

| 位置 | LTX-2 VAE 方案 (已放弃) | 问题 |
|---|---|---|
| Encoder | Audio VAE, 128d, 1 layer | generative model, 缺乏 phoneme discriminability |
| Adapter | 不存在 | — |
| Flatten | 5×1×128 = 640 | 信息量不足 (73x 降低) |
| proj1 | shape mismatch, 被 skip | pretrained weights 完全丢失 |

---

## 三、图中需要标注的参数统计

### 参数量分布

```
Omnilingual wav2vec2 encoder:  ~300M  🔒 frozen (不占训练显存)
Adapter (Linear 1024→768):    ~0.79M 🔥 随机初始化, 训练
AudioProjModel:
  proj1 (46080→512):          ~23.6M ✅ pretrained, fine-tune
  proj2 (512→512):            ~0.26M ✅ pretrained, fine-tune  
  proj3 (512→24576):          ~12.6M ✅ pretrained, fine-tune
  conv1 (24576→24576, k=2):   ~1.2B  ✅ pretrained, fine-tune
  norm (768):                 ~0.002M ✅ pretrained, fine-tune
DiT backbone:                 ~数 B   🔒 frozen / partially fine-tuned
Cross-attention layers:       包含在 DiT 中  ✅ pretrained
```

### 训练时梯度流

```
🔒 wav2vec encoder  →  无梯度 (torch.no_grad + frozen)
🔥 adapter          →  有梯度 (随机初始化)
🔥 AudioProjModel   →  有梯度 (pretrained weights fine-tune)
🔥 Cross-attention  →  有梯度 (pretrained weights fine-tune)
🔒 DiT backbone     →  看配置 (通常 frozen 或 LoRA)
```

---

## 四、图的视觉建议

### 颜色编码

- 🔵 蓝色: frozen 模块 (wav2vec encoder, DiT backbone)
- 🟢 绿色: pretrained weights + fine-tune (proj1/2/3, conv1, cross-attention)
- 🟠 橙色/红色: 新增模块, 随机初始化 (adapter)
- ⬜ 白色/灰色: 纯数据操作, 无参数 (sliding window, flatten)

### 箭头上标注 shape

每个箭头上标注 tensor shape 变化，例如：

```
[wav2vec] --(T, 12, 1024)-→ [Window] --(B,F,5,12,1024)-→ [Adapter] --(B,F,5,12,768)-→ [Flatten] --(B*F, 46080)-→ ...
```

### 并行分支（可选）

如果图空间允许，可以画两条并行分支对比：
- 左边: 原版 wav2vec2-base 路径 (768d, 无 adapter)
- 右边: 新版 Omnilingual 路径 (1024d, 有 adapter)
- 在 flatten 之后汇合到同一个 proj1

---

## 五、其他输入分支（图中需要画但不详细展开）

除了音频，Hallo3 还有其他条件输入同时送进 DiT：

```
1. Text prompt → CLIP text encoder → text embeddings → cross-attention
2. Reference face image → FaceProjModel → face context tokens → cross-attention
3. Noisy video latent → DiT input
4. Timestep embedding → AdaLN modulation
```

这些分支在图中画出来但标记为"不变"，重点突出音频分支的改动。
