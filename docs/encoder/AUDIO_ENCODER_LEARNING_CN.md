# Audio Encoder 学习路径

> 核心原则：不要从论文/原理开始，从数据的 shape 变化开始。
> 每个模块只问三个问题：**输入什么 shape？输出什么 shape？为什么要这样变？**

---

## 一、学习顺序总览

```
提取脚本 (离线)          训练数据加载           模型内部投影          训练/推理
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  第一步       │    │  第二步       │    │  第三步       │    │  第四步       │
│  跑提取脚本   │ →  │  拆解窗口采样 │ →  │  AudioProj   │ →  │  完整训练步   │
│  看一条音频   │    │  理解时间对齐 │    │  维度压缩     │    │  端到端验证   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
  audio → (T,128)     (T,128)→(F,5,1,128)  (F,5,1,128)→(F',32,768)   全链路 shape 对齐
```

| 顺序 | 文件 | 核心问题 |
|------|------|----------|
| 1 | `extract_audio_emb_ltx_vae.py` / `reextract_audio_emb.py` | 音频怎么变成 (T, 128) 或 (T, 12, 768) 的？ |
| 2 | `data_video.py` Stage2_SFTDataset | .pt 文件怎么变成 (F, 5, 1, 128) 的？ |
| 3 | `transformer.py` AudioProjModel | (F, 5, 1, 128) 怎么变成 (F', 32, 768) 的？ |
| 4 | `diffusion_video.py` | audio embedding 怎么和视频 latent 合在一起的？ |

**先把 1→2→3 彻底搞清楚，再看 4。**

---

## 二、第一步：跑提取脚本，看懂一条音频的完整变换

### 新链路 (LTX-2 VAE)

```bash
python scripts/extract_audio_emb_ltx_vae.py \
    --audio_dir /path/to/one_audio_folder \
    --output_dir /tmp/test_emb \
    --checkpoint /path/to/ltx-2.5-audio-vae-bf16.safetensors \
    2>&1 | grep "[AUDIO_DEBUG]"
```

### 旧链路 (Wav2Vec)

```bash
python scripts/reextract_audio_emb.py \
    --data_dir /path/to/hallo3_data \
    --audio_dir /path/to/audios \
    --wav2vec_model_path pretrained_models/wav2vec/wav2vec2-base-960h \
    2>&1 | grep "[AUDIO_DEBUG]"
```

### 看什么

拿一条 **3 秒**的短音频，对照 debug 输出**手算验证**：

**新链路 (VAE)**:
```
3 秒 × 16000 Hz = 48000 samples
MelSpectrogram: ceil(48000 / 160) + 1 ≈ 301 帧
下采样 ÷4 → T_latent ≈ 75
最终: (75, 128)    ← 75 = 3秒 × 25fps，128 = 8ch × 16mel
```

**旧链路 (Wav2Vec)**:
```
3 秒 × 16000 Hz = 48000 samples
wav2vec CNN → T_cnn (约 149)
linear_interpolation → seq_len = ceil(48000 / 16000 × 25) = 75
最终: (75, 12, 768)  ← 75帧, 12层hidden, 768维
```

**这一步建立的直觉**：音频时长 → embedding 帧数的映射关系，两条链路殊途同归到 ~25fps。

---

## 三、第二步：用 Python 交互式拆解窗口采样

```python
import torch

# 加载刚提取的 embedding
emb = torch.load("/tmp/test_emb/xxx.pt")
print(f"原始 shape: {emb.shape}")     # (75, 128) 或 (75, 12, 768)
print(f"dtype: {emb.dtype}")
print(f"前10个值: {emb[0, :10]}")     # 感受数值范围

# ========== 模拟 Stage2_SFTDataset 的窗口采样 ==========

# 假设取 49 帧视频，frame_interval=1，从第 5 帧开始
audio_margin = 2
margin_indices = torch.arange(2 * audio_margin + 1) - audio_margin
print(f"margin_indices: {margin_indices}")  # [-2, -1, 0, 1, 2]

# 模拟视频帧索引
ori_indices = torch.arange(5, 5 + 49)  # [5, 6, 7, ..., 53]

# 窗口采样：每个视频帧取 ±2 帧音频
center_indices = ori_indices.unsqueeze(1) + margin_indices.unsqueeze(0)
print(f"center_indices shape: {center_indices.shape}")  # (49, 5)
print(f"center_indices 范围: [{center_indices.min()}, {center_indices.max()}]")  # [3, 55]

audio_tensor = emb[center_indices]
print(f"窗口采样后: {audio_tensor.shape}")
# VAE:     (49, 5, 128)
# Wav2Vec: (49, 5, 12, 768)

# VAE 路径需要补 blocks 维度
if audio_tensor.dim() == 3:
    audio_tensor = audio_tensor.unsqueeze(2)
    print(f"补 blocks 维度后: {audio_tensor.shape}")  # (49, 5, 1, 128)
```

**这一步理解的核心**：
- 为什么要 ±2 帧？→ 给模型提供音频的**时间上下文**，不只看当前帧
- `(49, 5, 1, 128)` 的四个维度分别是：**视频帧数、时间窗口、blocks层数、特征维度**

---

## 四、第三步：搞懂 AudioProjModel 的维度压缩

这是**最容易出 bug 的地方**。在纸上画：

```
输入: (B, 49, 5, 1, 128)           ← DataLoader collate 后
              │
              │  rearrange "bz f w b c -> (bz*f) w b c"
              ▼
       (B×49, 5, 1, 128)
              │
              │  flatten: view(batch, 5*1*128)
              ▼
       (B×49, 640)                   ← ★ input_dim = 5×1×128 = 640
              │
              │  proj1: Linear(640 → 512) + ReLU
              │  proj2: Linear(512 → 512) + ReLU
              │  proj3: Linear(512 → 32×768 = 24576) + reshape
              ▼
       (B×49, 32, 768)              ← 32 个 context tokens
              │
              │  rearrange 回 batch → (B, 49, 24576)
              │  Conv1d(stride=2) × 2 次 → 时间维度 ÷4
              ▼
       (B, ~13, 24576)              ← 49 帧压成 ~13 帧
              │
              │  rearrange + LayerNorm
              ▼
       (B, ~13, 32, 768)            ← 送入 Transformer cross-attention
```

### 关键陷阱

**`input_dim` 必须和数据对齐**，否则 Linear 层直接报错：

| 路径 | seq_len | blocks | channels | input_dim |
|------|---------|--------|----------|-----------|
| VAE | 5 | **1** | **128** | 5×1×128 = **640** |
| Wav2Vec | 5 | **12** | **768** | 5×12×768 = **46080** |

**两条路径不能混用同一个 AudioProjModel 权重。** 切换路径时必须检查配置文件中的 `blocks` 和 `channels` 参数。

### 交互式验证

```python
# 模拟 AudioProjModel 的输入检查
B, F, W, BLK, C = 1, 49, 5, 1, 128  # VAE 路径
input_dim = W * BLK * C
print(f"VAE input_dim = {W}×{BLK}×{C} = {input_dim}")     # 640

B, F, W, BLK, C = 1, 49, 5, 12, 768  # Wav2Vec 路径
input_dim = W * BLK * C
print(f"Wav2Vec input_dim = {W}×{BLK}×{C} = {input_dim}")  # 46080

# 确认你的配置文件里 AudioProjModel 的 input_dim 和上面哪个对得上
```

---

## 五、第四步：跑一个 training step 看完整链路

```bash
# 只跑 1 步训练，过滤 debug 输出
python train_video.py ... --train-iters 1 2>&1 | grep "[AUDIO_DEBUG]"
```

### 重点验证清单

| 检查项 | 期望值 (VAE) | 期望值 (Wav2Vec) |
|--------|-------------|-----------------|
| Dataset 输出 audio_emb | (49, 5, 1, 128) | (49, 5, 12, 768) |
| batch collate 后 | (B, 49, 5, 1, 128) | (B, 49, 5, 12, 768) |
| AudioProjModel flatten | (B×49, 640) | (B×49, 46080) |
| AudioProjModel 输出 | (B, ~13, 32, 768) | (B, ~13, 32, 768) |
| 和视频帧数的关系 | 49 帧视频 → ~13 帧音频 token | 同左 |

如果任何一步 shape 不对，训练会报 `RuntimeError: mat1 and mat2 shapes cannot be multiplied`，这时候看 debug print 就能精确定位是哪一步出了问题。

---

## 六、两条链路对照学习

```
                   新链路 (LTX-2 VAE)              旧链路 (Wav2Vec2)
                   ─────────────────              ─────────────────
提取入口           extract_audio_emb_ltx_vae.py    reextract_audio_emb.py
预处理类           ltx_audio_vae/ops.py            sgm/utils/audio_processor.py
                   AudioProcessor (mel谱)          AudioProcessor (wav2vec特征)
编码器             ltx_audio_vae/audio_vae.py      sgm/models/wav2vec.py
                   AudioEncoder (VAE卷积)          Wav2VecModel (Transformer)
核心变换           波形→mel谱→卷积下采样→latent     波形→CNN特征→插值→Transformer
输出格式           (T, 128)                        (T, 12, 768)
帧率               ~25fps (16000/160/4)            自定义 (linear_interpolation)
语言依赖           无 (纯声学特征)                  有 (英语预训练)
优势               多语言、压缩率高                 成熟稳定、社区广
配置差异           blocks=1, channels=128           blocks=12, channels=768
```

---

## 七、常见问题排查

### Q: 训练报 shape mismatch
→ 看 `[AUDIO_DEBUG] AudioProjModel.forward()` 的 flatten 那行，检查 `input_dim` 是否等于 `w×b×c`。

### Q: audio_emb 加载后 shape 不对
→ 看 `[AUDIO_DEBUG] Stage2_SFTDataset` 的 `audio_emb 原始` 那行，确认 .pt 文件是哪条链路生成的。

### Q: 音频帧数和视频帧数对不上
→ 看 `center_indices range` 是否超出 `audio_emb` 的长度（`IndexError`），需要确保 audio_emb 的 T >= video 最大帧号 + audio_margin。

### Q: 切换 VAE/Wav2Vec 后训练崩溃
→ 必须同时修改三处：
1. 提取脚本（生成不同格式的 .pt）
2. json 元数据中的 key（`vocals_emb_vae_all` vs `vocals_emb_base_all`）
3. AudioProjModel 的 `blocks` 和 `channels` 参数

### Q: 如何确认 debug print 全部生效
```bash
grep -rn "AUDIO_DEBUG" --include="*.py" | wc -l
# 预期: 116 条，覆盖 10 个文件
```
