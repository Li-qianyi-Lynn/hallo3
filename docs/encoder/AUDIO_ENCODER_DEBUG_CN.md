# Audio Encoder 数据流调试手册

> 所有 debug print 统一前缀 `[AUDIO_DEBUG]`，可用 `grep "[AUDIO_DEBUG]"` 过滤查看。

---

## 一、完整数据流结构图

```
═══════════════════════════════════════════════════════════════════════════
  PATH A: LTX-2 Audio VAE  (Stage 2 新方案，语言无关)
  最终输出: (T, 128)
═══════════════════════════════════════════════════════════════════════════

  ① Raw Audio (.wav/.m4a)
     │  librosa.load(sr=16000, mono=True)
     ▼
  (samples,)  float32
     │  unsqueeze → repeat(1,2,1)  mono→stereo
     ▼
  (1, 2, samples)  ── Audio(waveform, sr=16000)
     │
  ② AudioProcessor.waveform_to_mel()          [ops.py]
     │  MelSpectrogram(sr=16000, n_fft=1024, hop=160, n_mels=64)
     │  → log(clamp(mel, min=1e-5))
     │  → permute(0,1,3,2)
     ▼
  (1, 2, T_mel, 64)   log-mel spectrogram
     │                 T_mel = ceil(samples / 160) + 1
     │
  ③ AudioEncoder.forward(spectrogram)          [audio_vae.py]
     │
     │  conv_in: Conv2d(2→128, k=3, s=1)
     ▼
  (1, 128, T_mel, 64)
     │
     │  ┌─ Downsampling Path (ch_mult=(1,2,4)) ─────────────────┐
     │  │  Level 0: 2× ResNet(128→128)  + Downsample(÷2)        │
     │  │           → (1, 128, T_mel/2, 32)                      │
     │  │  Level 1: 2× ResNet(128→256)  + Downsample(÷2)        │
     │  │           → (1, 256, T_mel/4, 16)                      │
     │  │  Level 2: 2× ResNet(256→512)  (最后一级, 无下采样)      │
     │  │           → (1, 512, T_mel/4, 16)                      │
     │  └────────────────────────────────────────────────────────┘
     ▼
  (1, 512, T_mel/4, 16)
     │
     │  Mid Block: ResNet → Identity → ResNet
     ▼
  (1, 512, T_mel/4, 16)
     │
     │  norm_out → SiLU → conv_out: Conv2d(512→16, k=3, s=1)
     │  (double_z=True → 2 × z_channels = 16)
     ▼
  (1, 16, T_mel/4, 16)
     │
  ④ _normalize_latents()
     │  chunk(2, dim=1)[0]  取 means 丢弃 logvar
     ▼
  (1, 8, T_latent, 16)     z_channels=8, mel_bins→16
     │
     │  patchify: rearrange("b c t f -> b t (c f)")      [patchifier.py]
     ▼
  (1, T_latent, 128)        8 × 16 = 128
     │
     │  PerChannelStatistics.normalize (逐通道 mean/std)  [ops.py]
     │  unpatchify: rearrange("b t (c f) -> b c t f")    [patchifier.py]
     ▼
  (1, 8, T_latent, 16)
     │
  ⑤ 提取脚本 extract_embedding()              [extract_audio_emb_ltx_vae.py]
     │  rearrange("b c t f -> b t (c f)")
     │  squeeze(0).float().cpu()
     ▼
  ★ 保存为 .pt: (T_latent, 128)  float32
     T_latent ≈ duration × 25  (16000 / 160 / 4 = 25 fps)


═══════════════════════════════════════════════════════════════════════════
  PATH B: Wav2Vec2  (Legacy baseline, 旧链路)
  最终输出: (T, 12, 768)
═══════════════════════════════════════════════════════════════════════════

  ① Raw Audio (.wav/.m4a)
     │  librosa.load(sr=16000)                 [audio_processor.py]
     ▼
  (samples,) float32
     │
  ② Wav2Vec2FeatureExtractor                   [audio_processor.py]
     │  将原始波形转为模型输入格式 (归一化等)
     │  np.squeeze(.input_values)
     ▼
  (samples,) float64  — 注意这里是 numpy array
     │
     │  计算 seq_len = ceil(samples / 16000 * fps)
     │  如: 48000 samples, 25fps → seq_len=75
     │
     │  可选: padding 到 clip_length 的整数倍
     │  F.pad(audio_feature, (0, pad_amount))
     │  unsqueeze(0) → (1, samples)
     ▼
  (1, samples) float32, on device
     │
  ③ Wav2VecModel.forward()                     [wav2vec.py]
     │
     │  feature_extractor (7-layer 1D CNN)
     ▼
  (1, C_feat, T_cnn)   C_feat=512
     │  transpose(1,2)
     ▼
  (1, T_cnn, 512)
     │
     │  linear_interpolation: F.interpolate(size=seq_len, mode='linear')
     ▼
  (1, seq_len, 512)     — 对齐到视频帧率
     │
     │  feature_projection: Linear(512→768) + LayerNorm + Dropout
     ▼
  (1, seq_len, 768)
     │
     │  mask_hidden_states
     │  Transformer Encoder (12 layers, output_hidden_states=True)
     ▼
  hidden_states: 13 层 (含输入层), 每层 (1, seq_len, 768)
     │
  ④ 后处理                                     [audio_processor.py]
     │  stack(hidden_states[1:], dim=1)  — 取 12 层 encoder 输出
     ▼
  (1, 12, seq_len, 768)
     │  squeeze(0)
     ▼
  (12, seq_len, 768)
     │  rearrange("b s d -> s b d")
     ▼
  ★ 保存为 .pt: (seq_len, 12, 768) float32
     seq_len ≈ duration × fps (如 25)


═══════════════════════════════════════════════════════════════════════════
  训练时: Stage2_SFTDataset.__getitem__()       [data_video.py]
═══════════════════════════════════════════════════════════════════════════

  ⑥ 加载 .pt 文件
     │  key = "{audio_type}_emb_{model_scale}_{features}"
     │  如 "vocals_emb_vae_all" 或 "vocals_emb_base_all"
     ▼
  audio_emb: (T_total, 128) 或 (T_total, 12, 768)
     │
     │  Temporal Windowing (audio_margin=2):
     │  margin_indices = [-2, -1, 0, 1, 2]
     │  center_indices = video_frame_indices ± margin
     │  audio_tensor = audio_emb[center_indices]
     ▼
  (num_frames, 5, 128)      ← VAE
  (num_frames, 5, 12, 768)  ← Wav2Vec
     │
     │  VAE 路径: dim==3 → unsqueeze(2) 补 blocks 维度
     ▼
  ★ batch["audio_emb"]:
     (num_frames, 5, 1, 128)    ← VAE    送入 DataLoader
     (num_frames, 5, 12, 768)   ← Wav2Vec

  5% 概率: audio dropout → 全置零


═══════════════════════════════════════════════════════════════════════════
  训练 pipeline: shared_step → forward → loss_fn  [diffusion_video.py]
═══════════════════════════════════════════════════════════════════════════

  ⑦ shared_step(batch)
     │  batch["audio_emb"]: (B, num_frames, 5, 1, 128)  ← collate 后加 B
     │  → 直接传入 forward() → loss_fn()
     ▼
  loss_fn 内部使用 audio_emb 做 cross-attention conditioning


═══════════════════════════════════════════════════════════════════════════
  AudioProjModel.forward()                      [transformer.py]
  将 audio embedding 投影到 Transformer 隐空间
═══════════════════════════════════════════════════════════════════════════

  ⑧ 输入: (B, F, 5, 1, 128)
     │
     │  rearrange "bz f w b c -> (bz*f) w b c"
     ▼
  (B*F, 5, 1, 128)
     │
     │  flatten: (B*F, 640)        5 × 1 × 128 = 640
     │  proj1: Linear(640→512) + ReLU
     │  proj2: Linear(512→512) + ReLU
     │  proj3: Linear(512→24576) + reshape
     ▼
  (B*F, 32, 768)                  32 context tokens × 768 dim
     │
     │  rearrange → (B, F, 24576)
     │  Conv1d(stride=2) × 2 次  → 时间维度 ÷4
     │  (奇数帧: 保留第一帧, 其余 conv)
     ▼
  (B, F', 24576)                  F' ≈ F/4
     │
     │  rearrange → (B, F', 32, 768)
     │  LayerNorm(768)
     ▼
  ★ (B, F', 32, 768)  → 送入 Transformer cross-attention
```

---

## 二、各文件 Debug Print 清单

### 2.1 `sgm/models/ltx_audio_vae/audio_vae.py` — AudioEncoder

| 位置 | 打印内容 |
|------|----------|
| `forward()` 入口 | spectrogram shape, dtype |
| `conv_in` 后 | shape, Conv2d 参数 (in_ch→ch) |
| `_run_downsampling_path` 每层 | level, block_idx, shape |
| 每级下采样后 | level, shape |
| `mid_block` 后 | shape |
| `_finalize_output` 后 | shape, double_z, z_channels |
| `_normalize_latents` | chunk 前后 shape, AudioLatentShape 四元组 |
| patchify 后 | shape (b, t, c*f) |
| normalize 后 | shape |
| unpatchify 后 | shape (b, c, t, f) |
| 最终输出 | shape |
| `encode_audio()` | 输入 waveform shape/sr, mel shape, 最终 latent shape |

### 2.2 `sgm/models/ltx_audio_vae/ops.py` — AudioProcessor & PerChannelStatistics

| 位置 | 打印内容 |
|------|----------|
| `waveform_to_mel()` 入口 | waveform shape, dtype |
| MelSpectrogram 参数 | sr, n_fft, hop_length, n_mels |
| MelSpectrogram 输出 | shape (b, ch, n_mels, time) |
| log-mel | shape, min/max 值域 |
| permute 最终输出 | shape (b, ch, time, n_mels) |
| `normalize()` | 输入 shape, mean/std shape 及值域 |

### 2.3 `sgm/models/ltx_audio_vae/patchifier.py` — AudioPatchifier

| 位置 | 打印内容 |
|------|----------|
| `patchify()` 入口 | shape (b, c, t, f) |
| patchify 输出 | shape (b, t, c*f) |
| `unpatchify()` 入口 | shape, target channels/mel_bins |
| unpatchify 输出 | shape (b, c, t, f) |

### 2.4 `sgm/models/wav2vec.py` — Wav2VecModel

| 位置 | 打印内容 |
|------|----------|
| `forward()` 入口 | input_values shape, seq_len |
| feature_extractor 后 | shape |
| transpose 后 | shape |
| linear_interpolation 后 | shape |
| feature_projection 后 | hidden_states shape |
| encoder 后 | last_hidden_state shape, num_hidden_states, 每层 shape |
| `feature_extract()` | 同上前四步 |
| `encode()` | extract_features shape → projection → encoder |
| `linear_interpolation()` | 输入 shape, target seq_len, 输出 shape |

### 2.5 `sgm/models/transformer.py` — AudioProjModel

| 位置 | 打印内容 |
|------|----------|
| `forward()` 入口 | audio_embeds shape, dtype |
| 模型参数 | seq_len, blocks, channels, input_dim, intermediate_dim, context_tokens, output_dim |
| rearrange 后 | (bz*f, w, b, c) shape |
| flatten 后 | shape, 乘法验证 (w*b*c) |
| proj1+ReLU 后 | shape |
| proj2+ReLU 后 | shape |
| proj3+reshape 后 | shape (context_tokens, output_dim) |
| rearrange 回 batch | shape (bz, f, m*c) |
| 每次 Conv1d | iter 编号, 奇/偶帧, shape |
| 最终输出 | shape (B, F', 32, 768) |

### 2.6 `data_video.py` — Stage2_SFTDataset

| 位置 | 打印内容 |
|------|----------|
| 加载 audio_emb | key 名, 文件路径, 原始 shape/dtype |
| margin 参数 | audio_margin 值, margin_indices 列表 |
| 视频采样 | start, end, frame_interval, ori_indices 范围, num_frames |
| center_indices | shape, min/max 范围 |
| 窗口采样后 | audio_tensor shape, dim |
| unsqueeze 判断 | VAE 路径 (dim==3) 或 Wav2Vec 路径, 补维后 shape |
| dropout 触发 | 是否置零 |
| 最终 item | mp4/ref_image/face_emb/mask_ref/audio_emb 全部 shape |

> bbox 分支和 mask 分支各有一套完整的打印。

### 2.7 `diffusion_video.py` — SATVideoDiffusionEngine

| 位置 | 打印内容 |
|------|----------|
| `shared_step()` | batch 中 audio_emb 是否存在, shape/dtype; x (video) shape |
| `forward()` | x shape; audio_emb shape → 传入 loss_fn |
| `sample()` | audio_emb shape/dtype → 传入 sampler |

### 2.8 `scripts/extract_audio_emb_ltx_vae.py` — 新链路提取脚本

| 位置 | 打印内容 |
|------|----------|
| 加载音频 | 文件路径, samples 数, sr, duration |
| stereo 转换 | waveform shape (batch, channels, samples) |
| mel spectrogram | shape (batch, ch, time, n_mels) |
| encoder latent | shape (batch, z_ch, T_latent, mel_bins) |
| rearrange tokens | shape (batch, T_latent, 128) |
| 最终保存 | shape (T_latent, 128) |

### 2.9 `sgm/utils/audio_processor.py` — Wav2Vec 旧链路预处理入口

| 位置 | 打印内容 |
|------|----------|
| `preprocess()` 加载音频 | 文件路径, samples 数, sr, duration |
| wav2vec_feature_extractor 后 | shape, dtype |
| 计算 seq_len | 公式: ceil(len / sr * fps) = 结果 |
| padding (clip_length) | clip_length, pad_amount, seq_len 变化 |
| 送入 wav2vec | audio_feature shape, seq_len |
| only_last_features 分支 | True: (T, 768); False: stack 12层 → rearrange |
| stack hidden_states | 层数, shape |
| rearrange 后 | shape (T, 12, 768) |
| 最终输出 | audio_emb shape, audio_length |
| `get_embedding()` | 同上流程 (无 padding/clip_length 步骤) |

### 2.10 `scripts/reextract_audio_emb.py` — Wav2Vec 旧链路提取脚本

| 位置 | 打印内容 |
|------|----------|
| 处理入口 | video stem, fps, m4a 路径 |
| 最终保存 | audio_emb shape, 输出路径 |

---

## 三、使用方法

### 运行时过滤

```bash
# 提取 audio embedding 时
python scripts/extract_audio_emb_ltx_vae.py ... 2>&1 | grep "\[AUDIO_DEBUG\]"

# 训练时 (SLURM)
srun python train_video.py ... 2>&1 | grep "\[AUDIO_DEBUG\]"

# 只看某个模块
... | grep "\[AUDIO_DEBUG\] AudioProjModel"
... | grep "\[AUDIO_DEBUG\] Stage2_SFTDataset"
... | grep "\[AUDIO_DEBUG\] AudioEncoder"
```

### 调试完成后清除

```bash
# 预览将删除的行
grep -rn "\[AUDIO_DEBUG\]" --include="*.py"

# 一键删除所有 debug print
find . -name "*.py" -exec sed -i '' '/\[AUDIO_DEBUG\]/d' {} +

# 验证清除干净
grep -rn "AUDIO_DEBUG" --include="*.py"
```

---

## 四、关键维度速查表

| 阶段 | VAE 路径 | Wav2Vec 路径 |
|------|----------|-------------|
| 原始音频 | (samples,) @16kHz | (samples,) @16kHz |
| Mel / CNN 特征 | (1, 2, T_mel, 64) | (1, T_cnn, 512) |
| Encoder 输出 | (1, 8, T_latent, 16) | (1, seq_len, 768) × 12层 |
| 保存的 .pt | **(T, 128)** | **(T, 12, 768)** |
| 窗口采样后 | (F, 5, 128) | (F, 5, 12, 768) |
| 补 blocks 维度 | **(F, 5, 1, 128)** | (F, 5, 12, 768) |
| batch collate | (B, F, 5, 1, 128) | (B, F, 5, 12, 768) |
| AudioProjModel 输入 flatten | (B*F, **640**) | (B*F, **46080**) |
| AudioProjModel 输出 | **(B, F', 32, 768)** | **(B, F', 32, 768)** |

> `T_latent ≈ duration_sec × 25`，`F = max_num_frames` (如 49)，`F' ≈ F/4` (Conv1d stride=2 × 2次)
