# 2026-10-16：AudioProjModel 完整数据流详解

承接 [2026-10-16_encoder_replacement_analysis.md](2026-10-16_encoder_replacement_analysis.md)。

本文档逐步拆解音频信号从原始波形到驱动口型的完整流程，重点解释每一步的 shape 变化和 flatten 的作用。

---

## 一、全局流程概览

```
音频波形 (.wav)
    ↓  ① wav2vec 编码
12 层 × 768 维的语音特征 (T, 12, 768)
    ↓  ② 窗口采样
每帧取前后各 2 帧，形成 5 帧窗口 (F, 5, 12, 768)
    ↓  ③ flatten
把每帧的 [5帧窗口 × 12层 × 768维] 拍平成一个大向量 (F, 46080)
    ↓  ④ 三层 MLP 投影
生成 32 个 context tokens (F, 32, 768)
    ↓  ⑤ Conv1d 时间下采样
帧数减半两次 F → F' ≈ F/4
    ↓  ⑥ Cross-attention 注入 DiT
每层 transformer: 视频特征(Q) × 音频 tokens(K,V) → 驱动口型
```

---

## 二、逐步详解

### 步骤 ①：wav2vec 编码

**代码位置**：`sgm/utils/audio_processor.py:124-134`

```python
embeddings = self.audio_encoder(audio_feature, seq_len=seq_len, output_hidden_states=True)
# embeddings.hidden_states: tuple of 13 tensors (embedding层 + 12层 Transformer)
# 每层 shape: (1, T, 768)

audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
# 跳过第 0 层（embedding），取第 1-12 层
# → (T, 12, 768)
audio_emb = rearrange(audio_emb, "b s d -> s b d")
# → (T, 12, 768)
```

T 是音频帧数（按 25fps 换算），12 是 Transformer 层数，768 是每层隐藏维度。

**12 层分别学到了什么**：
- 第 1-4 层：声学/音素特征（/a/ 还是 /m/，嘴巴怎么动）
- 第 5-8 层：音节/韵律（节奏、重音）
- 第 9-12 层：语义（词义上下文）

全部拼在一起给了一个完整的"语音金字塔"。

---

### 步骤 ②：窗口采样

**代码位置**：`data_video.py`（训练时的 DataLoader）

对于视频的第 i 帧，取音频特征的第 `[i-2, i-1, i, i+1, i+2]` 帧（前后各 2 帧的滑动窗口）。

```
音频帧:    ... t-2  t-1   t   t+1  t+2 ...
                 \___|____|____|___/
                        窗口=5
```

**shape 变化**：

```
输入: audio_emb (T, 12, 768) — 整段音频
    ↓ 对每个视频帧 i，取 [i-2 : i+3] 的窗口
输出: (F, 5, 12, 768) — F 是视频帧数

进入 AudioProjModel 时 batch 化:
    (B, F, 5, 12, 768) — B 是 batch size
```

**为什么要取窗口**：单帧音频只有当前时刻的信息，但嘴型变化是连续的——发 /a/ 的前一帧嘴巴在张开，后一帧在闭合。5 帧窗口让模型看到口型运动的上下文。

---

### 步骤 ③：flatten（核心问题所在）

**代码位置**：`sgm/models/transformer.py:390-394`

```python
# 输入: (B, F, 5, 12, 768)
audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
# → (B*F, 5, 12, 768)  — 把 batch 和帧合并，每帧独立处理

audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)
# → (B*F, 5*12*768) = (B*F, 46080)  — 这就是 flatten
```

**flatten 做了什么**：

把一个 3D 结构 `(5帧窗口, 12层, 768维)` 拍平成一个 1D 向量 `(46080,)`。

```
flatten 前（有结构）:
┌──────────────────────────────────────┐
│ 帧 t-2:  [第1层 768维] [第2层 768维] ... [第12层 768维]  │
│ 帧 t-1:  [第1层 768维] [第2层 768维] ... [第12层 768维]  │
│ 帧 t  :  [第1层 768维] [第2层 768维] ... [第12层 768维]  │
│ 帧 t+1:  [第1层 768维] [第2层 768维] ... [第12层 768维]  │
│ 帧 t+2:  [第1层 768维] [第2层 768维] ... [第12层 768维]  │
└──────────────────────────────────────┘
共 5 × 12 × 768 = 46,080 个数字

flatten 后（一维向量）:
[帧t-2_层1_dim1, 帧t-2_层1_dim2, ..., 帧t+2_层12_dim768]
长度 = 46,080
```

**为什么要 flatten**：因为接下来是全连接层（Linear），全连接层的输入必须是一维向量。flatten 让 proj1 能用一个矩阵乘法同时看到所有时间帧、所有层、所有维度的信息，从中提取最相关的口型特征。

**代价**：丢失了"哪个数字来自哪一帧、哪一层"的结构信息。这个结构需要 proj1 的权重自己学出来——46,080 个权重里，哪些对应哪一帧的哪一层，全靠训练学到。这也是为什么 proj1 的预训练权重很有价值。

---

### 步骤 ④：三层 MLP 投影

Multi-Layer Perceptron（多层感知机），是最基础的 feedforward neural network（前馈神经网络）结构
**代码位置**：`sgm/models/transformer.py:397-404`

```python
audio_embeds = torch.relu(self.proj1(audio_embeds))
# (B*F, 46080) → (B*F, 512)  压缩：从 46080 维里提取 512 维精华

audio_embeds = torch.relu(self.proj2(audio_embeds))
# (B*F, 512) → (B*F, 512)  非线性变换：进一步提纯

context_tokens = self.proj3(audio_embeds).reshape(batch_size, self.context_tokens, self.output_dim)
# (B*F, 512) → (B*F, 24576) → reshape → (B*F, 32, 768)
# 展开：从 512 维生成 32 个 768 维的 context tokens
```

**shape 变化**：

```
(B*F, 46080)  — flatten 后的原始特征
     ↓ proj1: Linear(46080, 512) + ReLU
(B*F, 512)    — 压缩到 512 维
     ↓ proj2: Linear(512, 512) + ReLU
(B*F, 512)    — 非线性变换
     ↓ proj3: Linear(512, 32×768) + reshape
(B*F, 32, 768) — 32 个 context tokens，每个 768 维
```

**三层各自的角色**：
- **proj1（46080→512）**：大幅压缩，从海量输入中筛选最重要的口型相关特征。这一层学到的权重决定了"12 层特征里哪些层、5 帧窗口里哪些帧对口型最重要"
- **proj2（512→512）**：非线性变换，让特征组合更灵活（两层 Linear+ReLU 比一层能拟合更复杂的函数）
- **proj3（512→24576）**：展开成 32 个 context tokens，给 cross-attention 用。32 个 token 可以编码口型的不同方面（嘴唇开合、牙齿露出、舌头位置等）

---

### 步骤 ⑤：Conv1d 时间下采样

**代码位置**：`sgm/models/transformer.py:408-427`

```python
context_tokens = rearrange(context_tokens, "(bz f) m c -> bz f (m c)", f=video_length)
# (B*F, 32, 768) → (B, F, 32*768) = (B, F, 24576)

# 做 2 次 Conv1d(kernel=2, stride=2)，帧数每次减半
for conv_iter in range(2):
    context_tokens = self.conv1(context_tokens)  # F → F/2
# 最终: (B, F', 24576)，F' ≈ F/4

context_tokens = rearrange(context_tokens, "b f (m c) -> b f m c", m=32)
# → (B, F', 32, 768)
context_tokens = self.norm(context_tokens)  # LayerNorm
```

**为什么要时间下采样**：

视频是 25fps，音频帧也是 25fps，但 DiT 处理视频时用的是 latent frames（经过 VAE 时间压缩后帧数更少）。Conv1d 下采样让音频帧数和 DiT 的 latent 帧数对齐。

```
音频帧:    25 fps → 取 13 帧（举例）
Conv1d ×2: 13 → 7 → 4（向下取整，奇数帧特殊处理）
DiT latent: 约 4 帧
```

---

### 步骤 ⑥：Cross-attention 注入 DiT

**代码位置**：`dit_video_concat.py:576-587`

在 DiT 的每一层 transformer block 里：

```python
# 视频特征作为 Query
audio_attn_input = layer.audio_input_layernorm(img_hidden_states)
audio_attn_input = rearrange(audio_attn_input, "b (t n) d -> (b t) n d", t=video_length)
# → (B*T, N, 768)  N=每帧的 visual patch 数量

# 音频 tokens 作为 Key 和 Value
kwargs['encoder_outputs'] = audio  # (B*T, 32, 768)

# Cross-attention: 视频问"我这帧该长什么口型？"，音频答"应该这样"
audio_attn_output = layer.audio_attn(audio_attn_input, mask, **kwargs)

# 残差连接：把音频驱动的口型信息加回视频特征
img_hidden_states = img_hidden_states + audio_attn_output
```

**Cross-attention 的含义**：

```
Q (视频): "我是第 t 帧的第 n 个 patch（嘴巴区域），我该长什么样？"
K (音频): "第 t 帧的 32 个音频 token 里，和你最相关的是这几个"
V (音频): "相关的音频信息是：当前在发 /a/ 音，嘴巴应该张开"

Attention(Q, K, V) = softmax(QK^T/√d) × V
→ 视频 patch 获得了"这一帧嘴巴该怎么动"的指令
→ 残差连接后，视频特征被音频"调制"了
```

**每一层 DiT 都做一次**，所以音频信号在多个尺度上影响视频生成（低层控制细节纹理，高层控制整体结构）。

---

## 三、完整 shape 变化汇总

以 batch_size=1, 视频 13 帧为例：

```
① wav2vec 编码:
   波形 (1, 80000) → wav2vec → (T=13, 12, 768)

② 窗口采样:
   (13, 12, 768) + 窗口±2 → (1, 13, 5, 12, 768)

③ flatten:
   rearrange → (13, 5, 12, 768)
   view      → (13, 46080)        ← flatten 在这里

④ 三层 MLP:
   proj1     → (13, 512)          ← 压缩
   proj2     → (13, 512)          ← 非线性变换
   proj3     → (13, 32, 768)      ← 展开成 32 tokens

⑤ Conv1d 下采样:
   rearrange → (1, 13, 24576)
   conv1d ×2 → (1, 4, 24576)     ← 帧数 13→7→4
   rearrange → (1, 4, 32, 768)
   LayerNorm → (1, 4, 32, 768)

⑥ Cross-attention:
   Q: 视频 (4, N, 768)            ← N 个 visual patches
   K,V: 音频 (4, 32, 768)         ← 32 个 audio tokens
   → 每帧每个 patch 都查询音频信息
   → 输出加回视频特征（残差）
```

---

## 四、关键设计决策的 why

| 设计 | 为什么这么做 |
|---|---|
| 取 12 层而非只取最后 1 层 | 不同层编码不同级别的语音信息（音素/韵律/语义），全部拼在一起给最丰富的信号 |
| 5 帧窗口 | 口型变化是连续的，单帧不够，需要看前后上下文 |
| flatten 成一个大向量 | 让全连接层一次看到所有信息，自己学出最优的跨层跨帧组合方式 |
| proj1 大幅压缩（46080→512） | 46080 维大部分是冗余的，只有少数维度和口型相关，压缩是在做特征选择 |
| 32 个 context tokens | 比 1 个 token 更灵活，不同 token 可以编码口型的不同方面 |
| Conv1d 下采样 | 音频帧率（25fps）和 DiT latent 帧率不一致，需要对齐 |
| 每层 DiT 都做 cross-attention | 多尺度注入，低层管细节（嘴唇纹理），高层管结构（嘴巴形状） |

---

## 五、和 adapter 方案的关系

用 Omnilingual wav2vec2（1024 维）时，adapter 插在步骤 ③ flatten 之前：

```
② 窗口采样后: (B*F, 5, 12, 1024)  ← Omnilingual 输出 1024 维
        ↓ adapter: Linear(1024, 768)
   (B*F, 5, 12, 768)              ← 降回 768 维
        ↓ flatten
   (B*F, 46080)                    ← 和原版完全一致
        ↓ proj1 (预训练权重直接加载)
   ...后面全部不变
```

adapter 只做维度映射（1024→768），不改变"5帧×12层"的结构。flatten 之后的所有层（proj1/proj2/proj3/conv1/norm）都保持原版维度，预训练权重完整保留。


  - DiT backbone（扩散模型主体）：完整加载，冻结不训 ✓                                                              
  - Cross-attention 层（audio/face attention）：完整加载 ✓
  - FaceProjModel（人脸投影）：完整加载 ✓                                                                           
  - 其他所有模块：完整加载 ✓                                                                                        
  - AudioProjModel：proj2、proj3、conv1、norm 都正常加载 ✓，只有 proj1.weight 被 skip 了（因为 shape 不匹配）       
                                                                                                                    
  所以整个模型里，只有 proj1 这一层是从随机初始化开始训的，其余全部用的预训练权重。   