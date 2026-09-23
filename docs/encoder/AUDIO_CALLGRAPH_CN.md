# Audio 数据流：函数调用链路

> 从训练入口 `train_video.py` 到最终的 cross-attention，每一步调用了哪个文件的哪个函数。

---

## 一、完整调用链路图

```
train_video.py:260  forward_step(data_iterator, model, args, ...)
│
│   :267  batch = next(data_iterator)          ← 从 DataLoader 取一批数据
│   :271  batch[key] = batch[key].cuda()       ← 搬到 GPU
│   :288  broad_cast_batch(batch)              ← 多 GPU 广播
│   :290  loss, loss_dict = model.shared_step(batch)   ← 调 shared_step（把原始的视频像素、参考图都压缩成 latent 空间的表示，然后才送去算扩散 loss。）
│
│   batch = {"mp4": ..., "audio_emb": (1,49,5,1,128), "face_emb": ..., ...}
│
└──► diffusion_video.py:197  SATVideoDiffusionEngine.shared_step(batch)
    │
    │   x = batch["mp4"]                          → (1, 49, 3, 480, 720)
    │   x = encode_first_stage(x)                 → (1, 13, 16, 60, 90)  视频VAE压缩 CogVideoX 的 3D-VAE 编码器，把原始像素视频压缩成 latent
    │
    └──► diffusion_video.py:162  .forward(x, batch)
        │
        └──► sgm/modules/diffusionmodules/loss.py:76  VideoDiffusionLoss.__call__(network, ref_network, denoiser, conditioner, x, batch)
            │
            │   ┌─────────────────────────────────────────────────┐
            │   │ 关键一步: batch2model_keys                       │
            │   │                                                  │
            │   │ additional_model_inputs = {                      │
            │   │     key: batch[key]                              │
            │   │     for key in self.batch2model_keys ∩ batch     │
            │   │ }                                                │
            │   │                                                  │
            │   │ yaml 配置: batch2model_keys: [audio_emb, face_emb] │
            │   │ 所以: additional_model_inputs = {                │
            │   │     "audio_emb": (1,49,5,1,128),                 │
            │   │     "face_emb": (1, 512)                         │
            │   │ }                                                │
            │   └─────────────────────────────────────────────────┘
            │
            │   加噪声: noised_input = x * alpha + noise * sigma
            │
            └──► denoiser.py:25  Denoiser.forward(network, input, sigma, cond, **additional_model_inputs)
                │
                │   c_in, c_noise, c_out = scaling(sigma)
                │
                └──► dit_video_concat.py:842  DiffusionTransformer.forward(x, timesteps, context, **kwargs)
                    │
                    │   kwargs 里面有 audio_emb 和 face_emb
                    │
                    │   if self.add_audio_module:
                    │       kwargs["audio_emb"] = kwargs["audio_emb"].to(self.dtype)  # → bfloat16
                    │
                    └──► transformer.py:766  BaseTransformer.forward(..., audio_emb, face_emb, ...)
                        │
                        │   ┌──────────────────────────────────────────┐
                        │   │ ★ AudioProjModel 在这里被调用              │
                        │   │                                           │
                        │   │ audio_emb = self.audio_proj(audio_emb)    │
                        │   │   输入: (1, 49, 5, 1, 128)                │
                        │   │   输出: (1, 13, 32, 768)                  │
                        │   │                                           │
                        │   │ assert f == 13   ← 必须和视频latent对齐    │
                        │   │                                           │
                        │   │ audio_emb = rearrange(                    │
                        │   │     "b f m c → (b f) m c"                 │
                        │   │ )                                         │
                        │   │ → (13, 32, 768)                           │
                        │   └──────────────────────────────────────────┘
                        │
                        │   for i in range(42):  # 42 层 Transformer
                        │       args = [hidden_states, mask, audio_emb, face_emb]
                        │
                        └──► transformer.py:496  AdaLNMixin.layer_forward(hidden_states, mask, audio, face_emb, ...)
                            │
                            │   === 1. Self-Attention (视频帧内部) ===
                            │   attn_input = input_layernorm(hidden_states)
                            │   attn_output = attention(attn_input, mask)
                            │   hidden_states += attn_output
                            │
                            │   === 2. Text Cross-Attention ===
                            │   cross_input = cross_input_layernorm(hidden_states)
                            │   cross_output = cross_attn(cross_input, encoder_outputs=text_tokens)
                            │   hidden_states += cross_output
                            │
                            │   ┌──────────────────────────────────────────┐
                            │   │ === 3. ★ Audio Cross-Attention ===        │
                            │   │                                           │
                            │   │ audio_input = audio_input_layernorm(      │
                            │   │     img_hidden_states                     │
                            │   │ )                                         │
                            │   │                                           │
                            │   │ # 按帧拆分: (b, t×n, d) → (b×t, n, d)    │
                            │   │ audio_input = rearrange(                  │
                            │   │     "b (t n) d → (b t) n d"              │
                            │   │ )                                         │
                            │   │                                           │
                            │   │ # 视频 query × 音频 key/value             │
                            │   │ audio_output = audio_attn(                │
                            │   │     audio_input,        ← query: 视频特征 │
                            │   │     encoder_outputs=audio  ← k,v: 音频    │
                            │   │ )                                         │
                            │   │ → CrossAttention.forward()                │
                            │   │                                           │
                            │   │ # 拼回来                                  │
                            │   │ audio_output = rearrange(                 │
                            │   │     "(b t) n d → b (t n) d"              │
                            │   │ )                                         │
                            │   │                                           │
                            │   │ hidden_states += audio_output  ← 残差连接 │
                            │   └──────────────────────────────────────────┘
                            │
                            │   === 4. Face Cross-Attention (类似 audio) ===
                            │   ...
                            │
                            │   === 5. MLP (前馈网络) ===
                            │   mlp_input = post_layernorm(hidden_states)
                            │   hidden_states += mlp(mlp_input)
                            │
                            │   return hidden_states
```

---

## 二、核心文件一览（按调用顺序）

| 调用顺序 | 文件 | 核心函数 | 做了什么 |
|---------|------|---------|---------|
| 1 | `train_video.py:260` | `forward_step()` | 训练入口，:290 调 shared_step |
| 2 | `diffusion_video.py:194` | `shared_step()` | 视频 VAE 编码，组装 batch |
| 3 | `diffusion_video.py:162` | `forward()` | 调 loss_fn |
| 4 | `loss.py:76` | `VideoDiffusionLoss.__call__()` | **从 batch 提取 audio_emb** (batch2model_keys) |
| 5 | `loss.py:124` | ↳ 调 denoiser | 加噪声，传 audio_emb 给 denoiser |
| 6 | `denoiser.py:39` | `Denoiser.forward()` | 透传 audio_emb 给 network |
| 7 | `dit_video_concat.py:874` | `DiffusionTransformer.forward()` | audio_emb 转 dtype，调 super |
| 8 | `transformer.py:807` | `BaseTransformer.forward()` | **调 AudioProjModel** (49帧→13帧) |
| 9 | `transformer.py:380` | `AudioProjModel.forward()` | MLP + Conv1d 投影 |
| 10 | `transformer.py:916` | ↳ 42层 layer loop | 传 audio_emb 给每层 |
| 11 | `transformer.py:577` | `AdaLNMixin.layer_forward()` | **audio cross-attention** |
| 12 | `transformer.py:220` | `CrossAttention.forward()` | 视频 query × 音频 key/value |

---

## 三、audio_emb 的变量名追踪

同一份数据，在不同文件里叫不同的名字：

```
data_video.py      → item["audio_emb"]           (49, 5, 1, 128)
                       ↓ DataLoader collate
train_video.py     → batch["audio_emb"]           (1, 49, 5, 1, 128)
                       ↓
diffusion_video.py → batch["audio_emb"]           (1, 49, 5, 1, 128)
                       ↓
loss.py            → additional_model_inputs["audio_emb"]
                       ↓ **kwargs 解包
denoiser.py        → **additional_model_inputs    
                       ↓ **kwargs 解包
dit_video_concat.py→ kwargs["audio_emb"]           .to(dtype)
                       ↓
transformer.py     → audio_emb (函数参数)           (1, 49, 5, 1, 128)
                       ↓ AudioProjModel
transformer.py     → audio_emb                     (13, 32, 768)
                       ↓ 传入每层
transformer.py     → audio (layer_forward 参数)     (13, 32, 768)
                       ↓ 送入 CrossAttention
transformer.py     → encoder_outputs               (13, 32, 768)
```

---

## 四、关键配置在哪里控制

```yaml
# configs/cogvideox_5b_i2v_s2.yaml

network_config:
  params:
    add_audio_module: True          ← 开关: True=Stage2有音频, False=Stage1无音频

loss_fn_config:
  params:
    batch2model_keys:
      - audio_emb                   ← 把 batch 里的 audio_emb 传给模型
      - face_emb                    ← 同时传 face_emb

# AudioProjModel 的参数在 transformer.py 里硬编码:
#   seq_len=5, blocks=1, channels=128  → input_dim=640
#   intermediate_dim=512, context_tokens=32, output_dim=768
```

---

## 五、如果你要改东西，改哪里

| 想改什么 | 改哪个文件 |
|---------|-----------|
| 音频提取方式（VAE/Wav2Vec） | `scripts/extract_audio_emb_ltx_vae.py` 或 `scripts/reextract_audio_emb.py` |
| 时间窗口大小（±2 帧） | `data_video.py` → `audio_margin` 参数 |
| 音频 dropout 概率（5%） | `data_video.py` → `random.random() < 0.05` |
| AudioProjModel 结构 | `transformer.py` → `AudioProjModel` 类 |
| 是否启用音频模块 | `configs/cogvideox_5b_i2v_s2.yaml` → `add_audio_module` |
| 传哪些 key 给模型 | `configs/cogvideox_5b_i2v_s2.yaml` → `batch2model_keys` |
| audio cross-attention 逻辑 | `transformer.py` → `AdaLNMixin.layer_forward` 577-585行 |
