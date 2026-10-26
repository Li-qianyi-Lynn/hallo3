# 2026-10-23：Hallo3 完整 Inference Workflow（所有 input/output 细节）

---

## 一、全局流程总览

```
┌─────────────────────── INPUTS ───────────────────────┐
│                                                       │
│  Text Prompt (string)                                 │
│  Reference Image (.jpg)                               │
│  Audio (.wav)                                         │
│                                                       │
└───────┬──────────────┬──────────────┬─────────────────┘
        ↓              ↓              ↓
   ┌─────────┐   ┌──────────┐   ┌──────────────┐
   │ T5 Text │   │ Image    │   │ Omnilingual  │
   │ Encoder │   │Processor │   │ wav2vec2     │
   │ (frozen)│   │+ FaceProj│   │ (frozen)     │
   └────┬────┘   └────┬─────┘   └──────┬───────┘
        ↓              ↓                ↓
   text_emb       face_emb +      audio_emb
   (1,77,768)     ref_latent      (T,12,1024)
        ↓              ↓                ↓
        │              │           ┌────────────┐
        │              │           │ Adapter    │
        │              │           │ +AudioProj │
        │              │           └─────┬──────┘
        │              │                 ↓
        │              │           audio_tokens
        │              │           (B,F',32,768)
        ↓              ↓                ↓
   ┌───────────────────────────────────────────┐
   │        DiT Diffusion Sampling Loop        │
   │  (VP-SDE, ~50 steps, CFG guidance)        │
   │                                            │
   │  randn noise → iterative denoising →       │
   │  denoised video latent                     │
   └─────────────────┬─────────────────────────┘
                     ↓
            ┌────────────────┐
            │ Video VAE      │
            │ Decoder        │
            └────────┬───────┘
                     ↓
            ┌────────────────┐
            │ Frame Assembly │
            │ + Audio Mux    │
            └────────┬───────┘
                     ↓
              OUTPUT: .mp4 视频
```

---

## 二、逐阶段详解

### Stage 1: Text Prompt Encoding

```
模块: FrozenT5Embedder (frozen)
文件: sgm/modules/encoders/modules.py

输入: text prompt string (e.g. "A person talking")
  ↓ T5 Tokenizer
  tokens: (1, 77)   ← max_length=77, padding="max_length"
  ↓ T5 Encoder (frozen)
输出: text_emb (1, 77, 768)

同时生成 unconditional embedding (空 prompt → 零向量)
用于 Classifier-Free Guidance (CFG)
```

### Stage 2: Reference Image Processing

分两个分支：face identity embedding 和 reference image latent。

#### 2a: Face Identity Embedding

```
模块: ImageProcessor (InsightFace)
文件: sgm/utils/image_processor.py

输入: reference image (.jpg), 任意分辨率
  ↓ InsightFace FaceAnalysis
  检测人脸, 选最大的 face bbox
  ↓ extract face embedding
输出: face_emb (512,) numpy → (1, 512) tensor on CUDA
```

#### 2b: Reference Image → Latent

```
输入: reference image (.jpg)
  ↓ resize to (480, 720), normalize to [-1, 1]
  image: (1, 3, 480, 720)
  ↓ unsqueeze 时间维度
  (1, 3, 1, 480, 720)
  ↓ 复制为 n_motion_frame=2 帧 + 加 mask
  mask_image: (1, 3, 3, 480, 720)  ← 1 ref + 2 motion frames
  ↓ Video VAE Encoder (first_stage_model.encode)
  (1, 16, 3, 60, 90)              ← C=16, H/8=60, W/8=90
  ↓ permute → pad to T=13 frames
输出: ref_latent (1, 13, 16, 60, 90)  ← 3 帧有内容 + 10 帧 zero padding

作为 cond["concat"] 送入 DiT
```

### Stage 3: Audio Processing（改版后的路径）

#### 3a: Omnilingual wav2vec2 Encoding

```
模块: Wav2VecModel (frozen)
文件: sgm/utils/audio_processor.py, sgm/models/wav2vec.py

输入: audio.wav, 16kHz, mono
  ↓ librosa.load → speech_array: (N_samples,)
  ↓ (可选) vocal separation via audio_separator
  ↓ Wav2Vec2FeatureExtractor
  audio_feature: (N_samples,)
  ↓ 计算 seq_len = ceil(N_samples / 16000 * 25)
  ↓ padding to clip_length 的整数倍
  ↓ wav2vec2 forward (output_hidden_states=True, torch.no_grad)
  hidden_states: 25 个 tensor (1 embedding + 24 transformer layers)
                 每层 (1, T, 1024)
  ↓ 取最后 12 层, stack + rearrange
输出: audio_emb (T, 12, 1024)
      T = 音频帧数 (按 25fps)
```

#### 3b: Sliding Window (process_audio_emb)

```
文件: sample_video.py:162-176

输入: audio_emb (T, 12, 1024)
  ↓ 对每帧 i, 取 [i-2, i-1, i, i+1, i+2] 窗口
  ↓ 边界处 clamp (不越界)
输出: audio_emb (T, 5, 12, 1024)
      5 = window size, 12 = layers, 1024 = dim
```

#### 3c: AudioProjModel（含 adapter）

```
模块: AudioProjModel
文件: sgm/models/transformer.py:348-433

输入: (B, F, 5, 12, 1024)   ← F = 当前 chunk 的帧数
  ↓ 🆕 Adapter: Linear(1024, 768)
  (B, F, 5, 12, 768)
  ↓ rearrange: (B*F, 5, 12, 768)
  ↓ flatten: (B*F, 46080)         ← 5 × 12 × 768
  ↓ proj1: Linear(46080, 512) + ReLU    ✅ pretrained
  (B*F, 512)
  ↓ proj2: Linear(512, 512) + ReLU      ✅ pretrained
  (B*F, 512)
  ↓ proj3: Linear(512, 32×768) + reshape ✅ pretrained
  (B*F, 32, 768)
  ↓ rearrange back: (B, F, 32×768)
  ↓ Conv1d(stride=2) × 2 次              ✅ pretrained
  (B, F', 32×768)   ← F' ≈ F/4
  ↓ rearrange + LayerNorm
输出: audio_tokens (B, F', 32, 768)
```

### Stage 4: Reference Feature Extraction

```
模块: ref_model (单独的 reference network)
文件: diffusion_video.py:326-337

输入: ref_image latent (1, 1, 16, 60, 90) + timestep=0 + uc text
  ↓ ref_model forward pass (不做 denoising, 只提取特征)
  ↓ 从每一层 transformer 提取 feature bank
输出: 每层 transformer 的 bank (cached features)
      复制到主模型 self.model 的对应层
      用于推理时的 reference attention
```

### Stage 5: Diffusion Sampling Loop

```
模块: VPSDEDPMPP2MSampler
文件: sgm/modules/diffusionmodules/sampling.py

初始化:
  x = randn(1, 13, 16, 60, 90)  ← 纯随机噪声
  num_steps ≈ 50
  guidance_scale ≈ 3.5-7.0

For step i in range(num_steps):
  ┌─────────────────────────────────────────────────────┐
  │ 1. 固定 reference frames (前 n_motion_frame 帧不变)    │
  │                                                       │
  │ 2. CFG: 同时跑 conditional 和 unconditional            │
  │    x_input = concat([x, x], dim=0)  ← batch=2        │
  │    cond_audio = concat([zeros, audio_tokens])          │
  │    cond_text  = concat([uc_text, c_text])              │
  │                                                       │
  │ 3. Denoiser forward:                                  │
  │    ↓ scaling: c_in, c_out, c_skip, c_noise            │
  │    ↓ DiT network forward:                             │
  │      input: x * c_in (1*2, 13, 16, 60, 90)           │
  │      + concat reference latent                        │
  │      + text cross-attention (T5 embeddings)           │
  │      + audio cross-attention (audio_tokens)           │
  │      + face identity conditioning (face_emb)          │
  │      + reference feature bank                         │
  │    ↓ output: predicted noise (2, 13, 16, 60, 90)      │
  │    ↓ CFG combine: out = uc + scale * (c - uc)         │
  │                                                       │
  │ 4. VPSDEDPMPP2M solver step                           │
  │    x = update(x, predicted_noise, alpha, sigma)       │
  └─────────────────────────────────────────────────────┘

输出: denoised latent (1, 13, 16, 60, 90)
```

### Stage 6: DiT Network 内部（单次 forward）

```
模块: BaseTransformer (DiT backbone)
文件: dit_video_concat.py

输入:
  - noisy_latent: (B, 13, 16, 60, 90) → patchify → (B, T*N, 768)
    T=13 frames, N=60*90/patch_size² patches per frame
  - text_emb: (B, 77, 768)
  - audio_tokens: (B, F', 32, 768)
  - face_emb: (B, 512) → FaceProjModel → face_tokens
  - ref_feature_bank: 每层缓存的 reference features
  - timestep → AdaLN modulation parameters

每一层 Transformer Block:
  ├── Self-Attention (spatial + temporal)
  │   img = img + self_attn(LayerNorm(img))
  │
  ├── Text Cross-Attention
  │   img = img + text_attn(LayerNorm(img), K=text_emb, V=text_emb)
  │
  ├── Face Cross-Attention
  │   img = img + face_attn(LayerNorm(img), K=face_tokens, V=face_tokens)
  │
  ├── 🔊 Audio Cross-Attention  ← 核心口型驱动
  │   Q = LayerNorm(img)
  │   Q = rearrange(Q, "b (t n) d -> (b t) n d")   per-frame
  │   K, V = audio_tokens  (B*T, 32, 768)
  │   audio_out = CrossAttention(Q, K, V)
  │   img = img + rearrange(audio_out, "(b t) n d -> b (t n) d")
  │
  ├── Reference Attention (from feature bank)
  │   img = img + ref_attn(img, K=bank, V=bank)
  │
  └── MLP (Feed-Forward Network)
      img = img + MLP(LayerNorm(img))

输出: predicted noise (B, T*N, 768) → unpatchify → (B, 13, 16, 60, 90)
```

### Stage 7: Video VAE Decoding

```
模块: VideoDecoder (first_stage_model)
文件: sgm/modules/autoencoding/temporal_ae.py

输入: denoised latent (1, 13, 16, 60, 90)
  ↓ un-scale: / scale_factor (≈ 0.18)
  ↓ chunk decoding (节省显存, 每次 decode 2-3 帧)
  Loop: 6 次 chunk decode
    chunk latent: (1, 16, 3, 60, 90)  ← 取 3 帧 latent
    ↓ VideoDecoder: 3D ConvTranspose + ResBlock + Temporal Attention
    chunk pixels: (1, 3, 3, 480, 720)  ← 3 帧 × 3 channels × 480 × 720
  ↓ concat all chunks along time
输出: video_pixels (1, 3, 13, 480, 720)
  ↓ permute → (1, 13, 3, 480, 720)
  ↓ denormalize: [-1, 1] → [0, 1]
  ↓ clamp
输出: video_frames (1, 13, 3, 480, 720) in [0, 1]
```

### Stage 8: Multi-Chunk Stitching（长视频）

```
文件: sample_video.py:358-437

长音频不能一次性处理, 分 chunk:
  L = (T-1)*4 + 1 = 49  ← 每个 chunk 的音频帧数
  n_motion_frame = 2     ← overlap 帧数

  For chunk t in range(num_chunks):
    1. 取音频 chunk: audio_emb[t*(L-2) : (t+1)*(L-2)]
    2. 拼接 overlap 帧: concat(pre_fix, audio_chunk)
    3. padding to L 帧
    4. 跑一次完整的 diffusion sampling + decoding → (1, T, 3, H, W)
    5. 去掉前 n_motion_frame 帧 (overlap 部分)
    6. 保存 motion frames 给下一个 chunk 用
       motion_image = samples[:, -2:]  ← 最后 2 帧
       ↓ encode → 作为下一个 chunk 的 reference latent

  最终: concat all chunks → (1, total_frames, 3, H, W)
  trim to 实际音频长度
```

### Stage 9: Video Assembly + Audio Muxing

```
文件: sample_video.py:79-131

输入: video_frames (1, total_frames, 3, H, W) in [0, 1]
  ↓ per frame:
    rearrange "c h w -> h w c"
    × 255 → uint8 [0, 255]
    (可选) crop padding
  ↓ imageio.get_writer(fps=25)
  写入 {i:06d}.mp4 (无音频)

  ↓ moviepy:
    VideoFileClip({i:06d}.mp4)
    AudioFileClip(audio_path)
    同步时长: 取 min(video_duration, audio_duration)
    video.set_audio(audio)
    write_videofile → {i:06d}_with_audio.mp4

输出: 最终 .mp4 文件 (视频 + 音频)
```

---

## 三、所有 Tensor Shape 汇总

### Inputs

| 数据 | 来源 | Shape | 说明 |
|---|---|---|---|
| text_prompt | 用户输入 | string | 文本描述 |
| ref_image | 用户输入 | (H, W, 3) uint8 | 参考人脸图片 |
| audio | 用户输入 | (N_samples,) float | 16kHz mono WAV |

### Encoding Outputs

| 数据 | 模块 | Shape | 说明 |
|---|---|---|---|
| text_emb | T5 Encoder | (1, 77, 768) | frozen text features |
| face_emb | InsightFace | (1, 512) | face identity vector |
| ref_latent | Video VAE Encoder | (1, 13, 16, 60, 90) | reference image in latent space |
| audio_emb (raw) | Omnilingual wav2vec2 | (T, 12, 1024) | 12 layers × 1024 dim per frame |
| audio_emb (windowed) | process_audio_emb | (T, 5, 12, 1024) | 5-frame sliding window |
| audio_tokens | AudioProjModel | (B, F', 32, 768) | 32 context tokens, F'≈F/4 |

### Diffusion Sampling

| 数据 | 阶段 | Shape | 说明 |
|---|---|---|---|
| initial noise | 初始化 | (1, 13, 16, 60, 90) | randn, float32 |
| noisy latent | 每步输入 | (2, 13, 16, 60, 90) | ×2 for CFG (cond + uncond) |
| predicted noise | DiT 输出 | (2, 13, 16, 60, 90) | noise prediction |
| CFG result | guidance | (1, 13, 16, 60, 90) | uc + scale*(c - uc) |
| denoised latent | 最终 | (1, 13, 16, 60, 90) | 去噪后的 video latent |

### Decoding & Output

| 数据 | 模块 | Shape | 说明 |
|---|---|---|---|
| un-scaled latent | / scale_factor | (1, 16, 13, 60, 90) | permuted for decoder |
| decoded chunk | VideoDecoder | (1, 3, 3, 480, 720) | 每次 decode 3 帧 |
| full video | concat chunks | (1, 13, 3, 480, 720) | 完整视频 |
| final frames | denormalize | (1, 13, 3, 480, 720) | [0, 1] float |
| output pixels | × 255 | (13, 480, 720, 3) | uint8 [0, 255] |
| output file | imageio + moviepy | .mp4 | 25fps, with audio |

---

## 四、DiT 内部每层的条件注入顺序

```
每一层 Transformer Block 的处理顺序:

img_hidden_states ─────────────────────────────────────────→
    │                                                        │
    ↓ 1. AdaLN Modulation (timestep conditioning)            │
    ↓ 2. Self-Attention (spatial + temporal)                  │
    ↓    + residual connection                                │
    ↓ 3. Text Cross-Attention (Q=img, K,V=text_emb)          │
    ↓    + residual connection                                │
    ↓ 4. Face Cross-Attention (Q=img, K,V=face_tokens)       │
    ↓    + residual connection                                │
    ↓ 5. Audio Cross-Attention (Q=img, K,V=audio_tokens) 🔊  │
    ↓    + residual connection                                │
    ↓ 6. Reference Attention (Q=img, K,V=ref_bank)           │
    ↓    + residual connection                                │
    ↓ 7. MLP / Feed-Forward Network                          │
    ↓    + residual connection                                │
    ↓                                                        │
img_hidden_states (updated) ────────────────────────────────→
```

每层都注入一次, 共 N 层 (DiT 的层数)
→ 多尺度条件注入: 低层管纹理细节, 高层管整体结构

---

## 五、关键配置参数

| 参数 | 值 | 来源 |
|---|---|---|
| sampling_num_frames (T) | 13 | inference.yaml |
| sampling_image_size | [480, 720] | inference.yaml |
| latent_channels (C) | 16 | model config |
| downsampling_factor (F) | 8 | Video VAE |
| latent_spatial | (60, 90) | 480/8, 720/8 |
| fps | 25 | inference.yaml |
| sample_rate | 16000 | audio config |
| num_denoising_steps | ~50 | sampler config |
| guidance_scale | 3.5-7.0 | guider config |
| n_motion_frame | 2 | multi-chunk overlap |
| audio_window_size | 5 | AudioProjModel.seq_len |
| audio_context_tokens | 32 | AudioProjModel.context_tokens |
| T5_max_length | 77 | text encoder |
