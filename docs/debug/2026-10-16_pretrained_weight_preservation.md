# 2026-10-16：保留预训练权重的 encoder 替换策略

承接 [2026-10-16_encoder_candidate_analysis.md](2026-10-16_encoder_candidate_analysis.md)。

## 一、问题

将 wav2vec2-base（768 维）替换为 MMS/Omnilingual wav2vec2（1024 维）后，AudioProjModel 的 `proj1.weight` 出现 shape mismatch：

| | 原版 checkpoint | MMS 新模型 |
|---|---|---|
| channels | 768 | 1024 |
| blocks | 12 | 12 |
| seq_len | 5 | 5 |
| proj1.weight shape | (512, **46080**) | (512, **61440**) |

当前的 monkey-patch 做法是**跳过不匹配的 key**，导致 AudioProjModel 完全从随机初始化开始训。这浪费了原版 Hallo3 预训练 checkpoint 里 AudioProjModel 已经学到的投影模式（哪些层重要、时序窗口内的注意力分布等）。

---

## 二、两种解决方向

### 方向 1：加 adapter 层，让 AudioProjModel 维度不变（推荐）

在 MMS 输出和 AudioProjModel 之间插入一个 `Linear(1024, 768)` 适配层：

```
MMS encoder (12 layers × 1024 dim)
   ↓
adapter: Linear(1024, 768)   ← 只有这一层随机初始化，参数量 1024×768+768 ≈ 79 万
   ↓
AudioProjModel(blocks=12, channels=768)   ← 完整加载预训练权重，无 shape mismatch
   input_dim = 5 × 12 × 768 = 46,080（和原版完全一致）
   ↓
proj1(46080→512) → proj2(512→512) → proj3(512→32×768)   ← 全部用预训练权重
   ↓
32 context tokens × 768 dim → cross-attention
```

**优势：**

- AudioProjModel 的 proj1 / proj2 / proj3 / conv1 / norm **全部保留预训练权重**
- 只有 adapter 层（~79 万参数）从随机初始化开始学
- 预训练的 AudioProjModel 已经学会了"从 12 层 × 768 维特征中提取口型驱动信号"这件事，这个能力直接迁移
- monkey-patch 不再需要跳过任何 key，checkpoint 加载零 warning

**代码改动：**

1. `audio_processor.py`：在 `preprocess()` 和 `get_embedding()` 中，对每层 hidden state 做 `adapter(x)` 降维（1024→768）
2. `transformer.py`：AudioProjModel 参数恢复为 `blocks=12, channels=768`（和原版一致）
3. adapter 层可以放在 `AudioProcessor` 类里，或作为 `AudioProjModel` 的新子模块

**adapter 位置选项：**

| 放在哪 | 优劣 |
|---|---|
| `AudioProcessor` 里（推理时） | 简单，但 adapter 不参与训练梯度（因为 `torch.no_grad()`） |
| `AudioProjModel` 里（训练时） | adapter 参与梯度更新，能端到端学习最优映射 |
| 数据预处理阶段（离线） | 最省事但 adapter 固定不更新，效果可能差 |

**推荐放在 `AudioProjModel` 里**，这样 adapter 跟着训练一起优化。

---

### 方向 2：reshape + 插值老权重（不推荐）

把原版 `proj1.weight` 从 (512, 46080) reshape 成 (512, 5, 12, 768)，沿 channel 维度插值到 (512, 5, 12, 1024)，再 flatten 回 (512, 61440)。

```python
# 伪代码
old_weight = checkpoint["audio_proj.proj1.weight"]  # (512, 46080)
old_weight = old_weight.view(512, 5, 12, 768)
new_weight = F.interpolate(old_weight, size=(12, 1024), mode='bilinear')
new_weight = new_weight.view(512, 61440)
```

**问题：**

- MMS（1024 维）和 wav2vec2-base（768 维）的特征空间不同，不是简单的"768 维 + 256 维扩展"
- 插值出来的权重在语义上未必有意义
- 仍然需要大量训练步数来矫正插值引入的偏差
- 实现复杂且难以验证正确性

---

## 三、结论

**采用方向 1（adapter 层）**，原因：

| 对比项 | 方向 1（adapter） | 方向 2（插值） | 当前做法（跳过） |
|---|---|---|---|
| 预训练权重保留 | **完整保留** | 部分保留（插值后偏移） | 完全丢失 |
| 新增随机参数 | ~79 万（adapter） | 0（但插值后需矫正） | ~2400 万（整个 AudioProjModel） |
| 收敛速度预期 | 快（只学 adapter） | 中（矫正插值偏差） | 慢（从零学投影） |
| 实现复杂度 | 低 | 高 | 最低 |

---

## 四、具体实现计划

在 `AudioProjModel.__init__()` 里加 adapter：

```python
class AudioProjModel(torch.nn.Module):
    def __init__(
        self,
        seq_len=5,
        blocks=12,
        channels=768,       # 保持原版维度
        encoder_channels=1024,  # MMS 的实际输出维度
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32
    ):
        super().__init__()
        # Adapter: 把 MMS 的 1024 维映射到原版的 768 维
        if encoder_channels != channels:
            self.adapter = torch.nn.Linear(encoder_channels, channels)
        else:
            self.adapter = None

        # 以下完全不变，保留预训练权重
        self.input_dim = seq_len * blocks * channels  # 46,080
        self.proj1 = torch.nn.Linear(self.input_dim, intermediate_dim)
        self.proj2 = torch.nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = torch.nn.Linear(intermediate_dim, context_tokens * output_dim)
        ...
```

在 `forward()` 里，flatten 之前对 channel 维度做适配：

```python
def forward(self, audio_embeds):
    # audio_embeds: (B, F, seq_len, blocks, encoder_channels)
    # 例如 (B, F, 5, 12, 1024)

    if self.adapter is not None:
        audio_embeds = self.adapter(audio_embeds)
        # → (B, F, 5, 12, 768)

    # 以下完全不变
    video_length = audio_embeds.shape[1]
    audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
    batch_size, window_size, blocks, channels = audio_embeds.shape
    audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)
    # → (bz*f, 46080)，和原版完全一致
    ...
```

**Checkpoint 加载时**：adapter 是新参数（key 为 `audio_proj.adapter.weight/bias`），checkpoint 里没有对应 key，`strict=False` 会自动跳过并随机初始化。proj1 / proj2 / proj3 / conv1 / norm 的 key 和 shape 完全匹配，正常加载。

---

## 五、需要改的文件

| 文件 | 改动 |
|---|---|
| `sgm/models/transformer.py` | AudioProjModel 加 `encoder_channels` 参数和 adapter 层，`channels` 恢复为 768 |
| `sgm/utils/audio_processor.py` | hidden states 取法：取最后 12 层，保持 1024 维不降维（adapter 在模型里做） |
| `train_video.py` | monkey-patch 可简化：只需处理 adapter 的新 key（missing key），不再有 shape mismatch |
