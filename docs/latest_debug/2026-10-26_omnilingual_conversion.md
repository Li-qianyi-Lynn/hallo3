# 2026-10-26：Omnilingual wav2vec2 转换与 G1 代码改动

承接 [2026-10-15_encoder_candidate_analysis.md](2026-10-15_encoder_candidate_analysis.md) 和 [2026-10-16_pretrained_weight_preservation.md](2026-10-16_pretrained_weight_preservation.md)。

---

## 一、Omnilingual wav2vec2 权重获取

### 1.1 结论：fairseq2 格式，需手动转换为 HuggingFace

Omnilingual wav2vec2 encoder-only 权重**存在**，但只有 fairseq2 格式（`.pt`），不在 HuggingFace 上。

| 模型 | 参数量 | 下载地址 |
|------|--------|---------|
| omniASR_W2V_300M | 315M | `https://dl.fbaipublicfiles.com/mms/omniASR-W2V-300M.pt` |
| omniASR_W2V_1B | 965M | `https://dl.fbaipublicfiles.com/mms/omniASR-W2V-1B.pt` |
| omniASR_W2V_3B | 3B | `https://dl.fbaipublicfiles.com/mms/omniASR-W2V-3B.pt` |

我们选用 **300M**（24 层 × 1024 维），和 MMS 300M 架构完全一致，但预训练数据更大（430 万小时 vs 50 万小时）、语言更多（1600+ vs 1100+）。

### 1.2 为什么不直接用 MMS

MMS `facebook/mms-1b-all` 虽然 HuggingFace 直接可用，但是 **48 层 × 1280 维、1B 参数**，远大于需要。而且 48 层存在 autoencoder 回归问题（见 1.4）。Omnilingual 300M 的 24 层更适合。

### 1.3 转换过程

1. 下载 checkpoint：

```bash
wget -4 -P /scratch/li_qiany_neu/pretrained_models/ \
    https://dl.fbaipublicfiles.com/mms/omniASR-W2V-300M.pt
```

2. fairseq2 → HuggingFace 转换：

```bash
python scripts/convert_omnilingual_to_hf.py \
    --input /scratch/li_qiany_neu/pretrained_models/omniASR-W2V-300M.pt \
    --output /scratch/li_qiany_neu/pretrained_models/omnilingual-w2v2-300m
```

3. 转换结果：

| 项 | 值 |
|---|---|
| checkpoint 总 key | 430 |
| 成功映射 | 422 |
| 跳过（预训练专用：quantizer, final_proj 等） | 8 |
| HF 模型总 key | 422 |
| 成功加载 | **420/422** |
| missing | 2（`feature_projection.layer_norm`，维度差异，不影响使用） |
| 验证加载 | 成功，315.4M 参数 |

转换后路径：`/scratch/li_qiany_neu/pretrained_models/omnilingual-w2v2-300m`

可直接用 `Wav2Vec2Model.from_pretrained()` 加载。

### 1.4 取层策略：24 层比 48 层更合适

根据 Pasad et al. (2021) 的 layer-wise probing 研究（arXiv: 2107.04734），wav2vec2 存在**自编码器回归现象**：

```
浅层 → 声学特征（频谱、基频）
中层 → 音素/语义峰值（最抽象、最有区分度）
深层 → 回归声学特征（像 autoencoder 的 decoder 端）
```

对于 MMS 48 层，最后 12 层（37-48）落在回归区，**不是最丰富的层**。
对于 Omnilingual 24 层，最后 12 层（13-24）覆盖中层+深层，信息更完整。

后续可做消融实验（A2）对比"最后 12 层" vs "均匀间隔 12 层"。

---

## 二、G1 代码改动总结

### 2.1 `sgm/models/transformer.py` — AudioProjModel 加 adapter ✅

```python
class AudioProjModel(torch.nn.Module):
    def __init__(self, ..., encoder_channels=1024):
        # 新增 adapter: Linear(1024, 768)
        if encoder_channels != channels:
            self.adapter = torch.nn.Linear(encoder_channels, channels)

    def forward(self, audio_embeds):
        # flatten 前调用 adapter: (B, F, 5, 12, 1024) → (B, F, 5, 12, 768)
        if self.adapter is not None:
            audio_embeds = self.adapter(audio_embeds)
        # 后续 flatten → proj1 → proj2 → proj3 维度完全不变
```

- 实例化改为 `AudioProjModel(encoder_channels=1024)`
- adapter ~79 万参数随机初始化；proj1/proj2/proj3/conv1/norm **全部保留预训练权重**
- checkpoint 加载时 `audio_proj.adapter.weight/bias` 是 missing key，自动随机初始化

### 2.2 `sgm/utils/audio_processor.py` — 取最后 12 层 ✅

```python
# 原版 wav2vec2-base 12 层：取全部 12 层
# Omnilingual 24 层：取最后 12 层（第 13-24 层）
all_hidden = embeddings.hidden_states[1:]  # 跳过 embedding 层
selected = all_hidden[num_layers - 12:]    # 最后 12 层
```

### 2.3 `data_preprocess.py` — 支持环境变量切换 encoder ✅

```python
wav2vec_model_path = os.environ.get(
    'WAV2VEC_MODEL_PATH',
    'pretrained_models/wav2vec/wav2vec2-base-960h'
)
```

### 2.4 `train_video.py` — 无需修改 ✅

monkey-patch 已有 shape mismatch 跳过逻辑。现在 proj1 shape 匹配（都是 512×46080），不会触发跳过。adapter 的新 key 自然是 missing key，`strict=False` 自动随机初始化。

---

## 三、关键发现：训练用预提取的 audio embedding

训练时 audio embedding **不是实时计算的**，而是预提取保存为 `.pt` 文件。数据集 JSON 通过 key `vocals_emb_base_all` 引用预提取的 embedding 路径。

```
data_video.py → torch.load(audio_emb_path) → (T, 12, 768) 或 (T, 12, 1024)
```

因此换 Omnilingual encoder 后，需要**重新预提取** 2702 条训练数据的 audio embedding：
- 新 shape：`(T, 12, 1024)`（之前原版是 `(T, 12, 768)`）
- adapter 在模型里做（训练时），不在预提取时做

---

## 四、待做

| # | 任务 | 状态 |
|---|------|------|
| 1 | ~~下载 Omnilingual 300M checkpoint~~ | ✅ |
| 2 | ~~fairseq2 → HuggingFace 转换~~ | ✅ 420/422 keys |
| 3 | 用新 encoder 重新预提取 2702 条 audio embedding | ⬜ |
| 4 | 更新训练 JSON 的 embedding 路径 | ⬜ |
| 5 | 开始 G1 训练（7000 步） | ⬜ |
| 6 | B1 val200 推理（hallo3 checkpoint 已下载） | ⬜ |

---

## 五、集群文件位置

| 文件 | 路径 |
|------|------|
| fairseq2 原始 checkpoint | `/scratch/li_qiany_neu/pretrained_models/omniASR-W2V-300M.pt` |
| 转换后 HF 模型 | `/scratch/li_qiany_neu/pretrained_models/omnilingual-w2v2-300m/` |
| 转换脚本 | `scripts/convert_omnilingual_to_hf.py` |
| G1 代码分支 | `train8`，工作目录 `~/hallo3` |
| B1 原版代码 | `main`，工作目录 `~/hallo3_orig` |
