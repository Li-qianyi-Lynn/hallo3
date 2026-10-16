# 2026-10-16：替换 encoder 候选方案分析

承接 [2026-10-16_encoder_replacement_analysis.md](2026-10-16_encoder_replacement_analysis.md)。

## 一、需求

- **多语言**：研究方向是小语种，encoder 必须覆盖尽可能多的语言
- **口型驱动**：特征需要有音素级区分度，能区分 /a/ vs /m/ 等
- **与 Hallo3 兼容**：尽量减少 AudioProjModel 和训练流程的改动
- **最新**：优先选用 2025-2026 年的模型
- **开源**：代码和权重必须公开可用

---

## 二、候选 Encoder 对比

| Encoder | 发布时间 | 语言覆盖 | 架构 | 预训练数据 | 特征维度 | 参数量 | 开源 | 与 Hallo3 兼容性 |
|---|---|---|---|---|---|---|---|---|
| **Omnilingual wav2vec2** | 2025.11 | **1600+ 语言** | wav2vec2 | 430 万小时 | 多层 x 1024 维 | 300M / 1B / 7B | **是** (GitHub + HF) | **最高** |
| **MMS** | 2023.05 | 1100+ 语言 | wav2vec2 | 50 万小时, 1400+ 语言 | 24 层 x 1024 维 | 1B | 是 (HF) | 最高 |
| **w2v-BERT 2.0** | 2023 | 143 语言 | Conformer | 450 万小时 | 24 层 x 1024 维 | 600M | 是 | 高 |
| **Whisper large-v3** | 2023.11 | ~99 语言 | Transformer enc-dec | 68 万小时 | 32 层 x 1280 维 | 1.5B | 是 | 中 |
| **CUPE** | 2025.08 | 多语言 (7语言验证) | 轻量级 | 多语言 | 定制 | 极小 | 是 (GitHub + HF) | 低 |
| **PASE** | 2025.04 | 未明确（英语为主） | 专为 lip sync | phoneme-viseme | 定制 | 未公开 | 未明确 | 低 |

---

## 三、推荐排序

### 第一选择：Omnilingual wav2vec2（Meta, 2025.11）

**最新、最大语言覆盖、完全开源、架构直接兼容。**

Omnilingual ASR 是 Meta 2025 年 11 月发布的，是 MMS 的后继项目。核心组件是 **Omnilingual wav2vec 2.0**——在 430 万小时、1600+ 语言上预训练的自监督语音表征模型。

**理由：**

1. **最新最强**：2025 年 11 月发布，比 MMS（2023）新两年，预训练数据从 50 万小时扩展到 430 万小时，语言从 1100+ 扩展到 1600+
2. **多种规模可选**：300M（轻量级）/ 1B / 7B（最高精度），可根据 GPU 资源选择
3. **完全开源**：
   - GitHub: [facebookresearch/omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr)
   - HuggingFace: [facebook/omniASR-LLM-7B](https://huggingface.co/facebook/omniASR-LLM-7B)
4. **与 Hallo3 架构直接兼容**：底层仍是 wav2vec2，输出为多层 Transformer hidden states，和原版 Hallo3 的 wav2vec 接口完全一致
5. **研究新颖性**：2025 年底才发布，在 talking head 领域尚无人用过

**代码改动预估：**

```python
# wav2vec.py：换 model ID
# 原版
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
# 768 dim, 12 layers

# 换成 Omnilingual wav2vec2（需确认 HF 上的 encoder-only 模型 ID）
model = Wav2Vec2Model.from_pretrained("facebook/omnilingual-w2v2-300m")  # 待确认
# 1024 dim, 24+ layers
```

```python
# transformer.py AudioProjModel：调维度
AudioProjModel(seq_len=5, blocks=12, channels=1024)  # 取 12 层
# input_dim = 5 * 12 * 1024 = 61,440（比原版 46,080 更丰富）
```

**待确认：** Omnilingual wav2vec2 的 encoder-only 权重是否单独发布在 HuggingFace 上（目前看到的是完整 ASR 系统的权重，需要确认能否只取 encoder 部分的 hidden states）

**参考文献：**
- [Omnilingual ASR: Open-Source Multilingual Speech Recognition for 1600+ Languages (Meta, 2025.11)](https://arxiv.org/pdf/2511.09690)
- [GitHub: facebookresearch/omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr)
- [Meta AI Blog](https://ai.meta.com/blog/omnilingual-asr-advancing-automatic-speech-recognition/)

---

### 第二选择：MMS（facebook/mms-1b-all, 2023）

如果 Omnilingual wav2vec2 的 encoder-only 权重不方便单独提取，MMS 是成熟的退路。

**理由：**

1. **代码和权重完全开源**：HuggingFace 上直接 `Wav2Vec2Model.from_pretrained("facebook/mms-1b-all")` 即可加载，[有完整文档](https://huggingface.co/docs/transformers/en/model_doc/mms)
2. **1100+ 语言**，仍然远超其他候选
3. **与 Hallo3 完全兼容**：同为 wav2vec2 架构，24 层 x 1024 维

**顾虑：** 2023 年发布，不够新。但作为 encoder 在 talking head 领域仍然没人用过。

**参考文献：**
- [Scaling Speech Technology to 1,000+ Languages (Meta, 2023)](https://arxiv.org/pdf/2305.13516)
- [HuggingFace MMS 文档](https://huggingface.co/docs/transformers/en/model_doc/mms)

---

### 第三选择：Whisper encoder

**理由：**

1. [MuseTalk](https://arxiv.org/html/2410.10122v1) 已验证 Whisper encoder 可以驱动 lip sync（用的是 Whisper-Tiny）
2. Whisper 训练包含大量多语言有标注数据（68 万小时），encoder 特征语义层次更高
3. ~99 语言覆盖

**顾虑：**

- Whisper 是 encoder-decoder 架构（为转录设计），encoder 特征可能更偏向"转录"而非"区分音素"
- 维度更大（large-v3: 32 层 x 1280 维），需要更多适配
- 语言覆盖只有 ~99 种，远不及 Omnilingual/MMS

**参考文献：**
- [MuseTalk: Real-Time High Quality Lip Synchronization with Latent Space Inpainting](https://arxiv.org/html/2410.10122v1)

---

### 第四选择：w2v-BERT 2.0

**理由：**

1. 在多语言 benchmark（ML-SUPERB 2025）上性能最强
2. 24 层 Conformer + 对比学习 + masked prediction，450 万小时预训练
3. 是 [SeamlessM4T](https://arxiv.org/pdf/2308.11596) 的 backbone

**顾虑：**

- 只覆盖 143 语言
- Conformer 架构和 wav2vec2 的 Transformer 有差异，适配代码更多

**参考文献：**
- [SeamlessM4T: Massively Multilingual & Multimodal Machine Translation](https://arxiv.org/pdf/2308.11596)

---

### 补充参考：CUPE（2025.08）

[CUPE (Contextless Universal Phoneme Encoder)](https://arxiv.org/pdf/2508.15316) 是一个轻量级音素编码器，120ms 窗口即可捕获关键音素特征，跨语言泛化能力强。开源在 [GitHub](https://github.com/tabahi/bournemouth-forced-aligner) 和 [HuggingFace](https://huggingface.co/Tabahi/CUPE-3-mls7) 上。

但它设计目标是 forced alignment，不是语音表征学习，特征维度较小，不适合直接替代 wav2vec。可以作为辅助模块（如提供显式的音素标签）而非主 encoder。

---

### 不推荐：PASE

[PASE (Phoneme-Aware Speech Encoder)](https://arxiv.org/abs/2504.05803) 专为 lip sync 设计（提升 13-14%），但多语言支持不明确，开源状态不明，适配成本高。

---

## 四、结论

**Omnilingual wav2vec2 是最优选择（如果 encoder 权重可单独提取），MMS 是成熟退路。**

| 维度 | Omnilingual wav2vec2 | MMS（退路） |
|---|---|---|
| 发布时间 | 2025.11（最新） | 2023.05 |
| 语言覆盖 | 1600+ | 1100+ |
| 预训练数据 | 430 万小时 | 50 万小时 |
| 架构兼容 | wav2vec2，直接兼容 | wav2vec2，直接兼容 |
| 开源 | GitHub + HF | HF |
| 工程改动 | 换 model ID + 调维度 | 换 model ID + 调维度 |
| 研究新颖性 | 极高（2025 年底发布） | 中（2023 年但 talking head 领域没人用过） |

---

## 五、下一步

1. **确认 Omnilingual wav2vec2 的 encoder-only 权重**：检查 [GitHub 仓库](https://github.com/facebookresearch/omnilingual-asr) 和 HuggingFace 上是否有 300M encoder-only 的 checkpoint，能否直接用 `Wav2Vec2Model` 加载
2. 如果可以：修改 `wav2vec.py` 加载 Omnilingual，调整 AudioProjModel 维度
3. 如果不方便提取：退回 MMS（`facebook/mms-1b-all`），加载方式已验证可用
4. 用同样 2702 条数据训同样步数，同样 5 条 val 打 SyncNet，和 LTX-2 VAE 对比
5. 如果效果好，再扩大数据量和步数做正式实验
