# Related Work Survey: Audio-Driven Talking Head Generation (2024–2026)

**调研日期**: 2026-10-29（2026-10-30 更新 2026 年文章）  
**目的**: 梳理 2025-2026 年顶会/顶刊中与本研究相关的工作，明确 novelty 定位

---

## 一、最直接相关（竞争/互补工作）

### 1.1 多语言 / 跨语言 Talking Head


| 论文                                  | 会议                 | 关键贡献                                                   | 与我们的关系                                                        |
| ----------------------------------- | ------------------ | ------------------------------------------------------ | ------------------------------------------------------------- |
| **MultiTalk**                       | Interspeech 2024   | 423h, 20 语言数据集 + 语言特定 style embedding，证明多语言数据显著提升跨语言效果 | 最直接竞品之一，但基于 3DMM/NeRF，非 DiT 架构                                |
| **MuEx** (Phoneme-Viseme Alignment) | arXiv 2025.10      | Phoneme-Guided MoE，用音素/视素作为跨语言桥梁，12 语言、95h，不需要语言标签     | 核心竞品，思路类似（语言无关中间表征），但用 phoneme-viseme 而非 Omnilingual wav2vec2 |
| **VQTalker**                        | arXiv 2024.12      | VQ 量化面部运动 token，针对非印欧语系泛化                              | 方向相关，离散运动表征 vs 我们的多语言自监督语音表征                                  |
| **Polyglot**                        | arXiv 2026.04      | 首个统一扩散模型做多语言语音驱动面部动画（3DMM-based），无需语言/说话人标签            | 方向相关，但基于 3DMM 而非 Video DiT                                    |
| **AudioFace**                       | arXiv 2026.05      | 用多模态 LLM 先验 + 音素级线索辅助 ARKit blendshape 预测，提升跨语言泛化      | 方向相关，language-assisted 音频编码器                                  |
| **TalkVid**                         | CVPR 2026 Findings | 1244h, 15 语言大规模数据集，跨语言评估 benchmark                     | 我们使用的数据集                                                      |


### 1.2 语言无关音频表征（编码器替换方向）


| 论文                       | 会议            | 关键贡献                                                                       | 与我们的关系                                                                                                                |
| ------------------------ | ------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **PASE**                 | arXiv 2025.04 | Phoneme-Aware Speech Encoder，解决音素-视素多对多歧义，lip sync 提升 13-14%，plug-and-play | 同样关注音频编码器改进，但方向不同（phoneme-aware vs language-agnostic）                                                                 |
| **MuseTalk**             | arXiv 2024    | 直接用 mel spectrogram 做 lip sync（不用 wav2vec2），VAE latent space inpainting    | 用途不同：MuseTalk 做嘴部区域 inpainting，mel 足够驱动局部修复；我们的 DiT cross-attention 需要驱动整个视频生成，G4 已证明 mel-based 表征在此架构下不可行（conf=1.04） |
| **PD-GS**                | arXiv 2026.08 | Phoneme-conditioned 3D Gaussian Splatting，改善音素-视素对齐                        | 音素级别音频编码 + 3DGS，另一种编码器改进路线                                                                                            |
| **From Tokens to Faces** | arXiv 2026.06 | 研究离散语音 token 表征（来自 speech LM）用于 3D 面部动画                                    | 音频表征研究，与我们的 wav2vec2 连续表征形成对比                                                                                         |


### 1.3 Hallo 系列后续


| 论文             | 会议            | 关键贡献                                     | 与我们的关系               |
| -------------- | ------------- | ---------------------------------------- | -------------------- |
| **Hallo2**     | ICLR 2025     | UNet backbone, patch-drop 长视频, 4K 分辨率    | 前代，UNet 方案           |
| **Hallo3**     | CVPR 2025     | CogVideoX-5B DiT + audio cross-attention | 我们的 base model       |
| **Hallo4**     | arXiv 2025.05 | DPO + 时间运动调制，提升生成质量                      | 后继，关注质量而非多语言，**不冲突** |
| **Hallo-Live** | ACM MM 2026   | 实时流式，异步双流 + 偏好蒸馏，20.38 FPS，2×H200        | 关注实时性，方向不同           |


---

## 二、同领域重要 Baseline

### 2.1 扩散模型 Talking Head（2024–2025）


| 论文                        | 会议                       | 关键贡献                                                      |
| ------------------------- | ------------------------ | --------------------------------------------------------- |
| **VASA-1**                | NeurIPS 2024             | 解耦面部潜空间，512×512@40fps，SOTA baseline                       |
| **Loopy**                 | ICLR 2025                | 纯音频驱动 + inter/intra-clip 时间模块，长时运动依赖                      |
| **FLOAT**                 | ICCV 2025                | Flow matching 替代 DDPM，运动潜空间                               |
| **OmniSync**              | NeurIPS 2025 (Spotlight) | DiT lip sync，无 mask，从加噪源视频直接编辑                            |
| **EmotiveTalk**           | CVPR 2025                | 音频解耦为情感流 + 内容流                                            |
| **Teller**                | CVPR 2025                | 首个自回归流式框架                                                 |
| **MoEE**                  | CVPR 2025                | Mixture of Emotion Experts                                |
| **FantasyTalking**        | arXiv 2025.04            | Wan2.1 DiT backbone + audio conditioning（类似我们用 CogVideoX） |
| **Long-Term TalkingFace** | ICML 2025                | Motion-prior 条件扩散，长视频                                     |
| **Sonic**                 | arXiv 2024               | SVD backbone + motion-decoupled control，lip sync +45%     |


### 2.2 扩散模型 Talking Head（2026）


| 论文                  | 会议            | 关键贡献                                                                             |
| ------------------- | ------------- | -------------------------------------------------------------------------------- |
| **READ**            | AAAI 2026     | Temporal VAE + Audio-to-Video DiT (A2V-DiT) + 异步噪声调度，实时 1:1 处理                   |
| **FantasyTalking2** | AAAI 2026     | TLPO（时间步-层自适应偏好优化），对齐多维人类偏好（运动自然度、lip sync、视觉质量）                                 |
| **EchoMimicV3**     | AAAI 2026     | 1.3B 参数统一多模态多任务人体动画                                                              |
| **AUHead**          | ICLR 2026     | 音频-语言大模型 + "emotion-then-AU" chain-of-thought，24 维 Action Unit 驱动情感 talking head |
| **PC-Talk**         | CVPR 2026     | Lip-audio alignment control (LAC) + emotion control (EMC)，隐式关键点变形                |
| **AudioAvatar**     | CVPR 2026     | Splat-based 全身 avatar，单照片生成                                                      |
| **SyncCache**       | ECCV 2026     | 训练无关扩散缓存，4.12× 加速，几乎无损 lip sync                                                  |
| **EchoAvatar**      | SIGGRAPH 2026 | 统一流式架构，层次化 token 损坏策略，实时全身运动生成                                                   |
| **HighSync**        | arXiv 2026.05 | 首个原生 512×512 lip sync 扩散模型                                                       |
| **Lip Forcing**     | arXiv 2026.06 | 14B 双向视频扩散蒸馏为 1.3B 因果模型，31 FPS 实时 lip sync                                       |
| **EditYourself**    | arXiv 2026.01 | 14B DiT video-to-video + audio cross-attention，对话编辑                              |


### 2.3 其他相关


| 论文              | 会议            | 关键贡献                           |
| --------------- | ------------- | ------------------------------ |
| **EchoMimic**   | AAAI 2025     | SD backbone + 可编辑 landmark 条件  |
| **MoDistalker** | AAAI 2025     | 运动解耦扩散模型                       |
| **Ditto**       | ACM MM 2025   | 运动空间扩散，实时                      |
| **MimicTalk**   | NeurIPS 2024  | Few-shot 3DGS 个性化 talking head |
| **LTX-2**       | arXiv 2025.01 | 14B 视频 + 5B 音频双流 DiT，联合音视频生成   |
| **KDTalker++**  | ACM MM 2026   | 关键点时空扩散 + 语音克隆 + 背景编辑          |


---

## 三、我们的 Novelty 定位

### 现有工作的空白

```
                    旧架构 (3DMM/NeRF/UNet)          新架构 (Video DiT)
                    ─────────────────────          ──────────────────
单语言 (英语)        VASA-1, EchoMimic...            Hallo3/4, OmniSync, READ, PC-Talk
多语言              MultiTalk, MuEx, Polyglot        ← 空白（我们的位置）
```

1. **在 DiT 架构上做多语言**：现有多语言工作（MultiTalk、MuEx、Polyglot）都基于旧架构，暂时未发现其他人在 video DiT 上做跨语言泛化
2. **Omnilingual wav2vec2 + adapter 替换英语 wav2vec2-base**：保留 Hallo3 预训练权重的同时接入多语言 encoder（1600+ 语言），用 Linear(1024→768) adapter 桥接维度差异。与 MuEx 的 phoneme-viseme 路线和 MuseTalk 的 mel 路线都不同
3. **消融实验（编码器 × 数据 2×2 矩阵）** 可以清晰拆分编码器和数据各自的贡献

### 为什么这个空白存在？

没人在 Video DiT 上做跨语言泛化，不是因为不重要，而是**条件刚刚成熟**：

1. **Video DiT 太新**：Hallo3（CVPR 2025）是最早用 Video DiT 做 talking head 的工作之一，架构刚出来一年多，大家还在解决基础问题（质量、长视频、实时性），还没关注多语言
2. **算力门槛高**：Video DiT 动辄 5B 参数，训练需要多卡 H100/H200。3DMM 方案轻量得多，多语言消融实验成本低，所以多语言研究自然先在 3DMM 上做了
3. **多语言数据刚出来**：TalkVid（CVPR 2026）、MultiTalk（Interspeech 2024）都很新，之前主流 benchmark（HDTF、VoxCeleb）基本是英语，没有动力做跨语言
4. **英语够用，问题被忽略**：wav2vec2-base 在英语 benchmark 上表现好，只有拿非英语音频去跑才会发现 lip sync 崩——但大多数团队不做这个测试
5. **两个社区不重叠**：做多语言语音的人（NLP/speech 社区）和做视频生成的人（CV 社区）交叉关注少

> 这个逻辑可以直接用在论文 Introduction 中 motivate 我们的工作。

### Related Work 写作时需重点讨论的论文

1. **MuEx** — 最近的跨语言方案，用 phoneme-viseme 做桥梁 vs 我们用 Omnilingual wav2vec2（自监督多语言表征），需要讨论两种 language-agnostic 路线的 trade-off
2. **Polyglot**（arXiv 2026.04）— 首个统一扩散模型做多语言面部动画，但基于 3DMM，非 Video DiT。需要说明我们的方案在 2D 视频质量上的优势
3. **AudioFace**（arXiv 2026.05）— 用多模态 LLM 辅助跨语言泛化，思路不同（LLM 先验 vs 多语言自监督 encoder），需要对比讨论
4. **MultiTalk** — 多语言数据集 + 3D 方案，我们在 2D DiT 上做类似的事
5. **Hallo4** — Hallo 系列最新，需要说明我们的工作与 Hallo4 正交（他们做质量，我们做多语言）
6. **READ**（AAAI 2026）— A2V-DiT 架构，和我们都用 DiT 做 talking head，但他们没做多语言
7. **MuseTalk** — 用 mel 做嘴部 inpainting，用途和架构都不同；我们的 G4 已证明 mel-based 表征在 DiT cross-attention 下不可行
8. **PASE** — phoneme-aware 编码器，另一种改进音频表征的路线

### 需要注意的风险

- **Polyglot**（arXiv 2026.04）已经做了多语言扩散，虽然是 3DMM-based 但方向重叠 → 需要强调 Video DiT 架构的差异性（端到端 2D 视频生成 vs 3DMM 参数预测 + 渲染），以及 Omnilingual wav2vec2 + adapter 的独特性。实际质量对比需要在同一评估集上定量验证
- **AudioFace**（arXiv 2026.05）用 LLM 做语言辅助，如果被顶会接收可能成为强竞品 → 需要讨论我们的方案更轻量（不依赖 LLM inference）
- **MuEx**（arXiv 2025.10）如果被顶会接收并发表在我们之前，novelty 会受挑战 → 需要强调 DiT backbone 和消融实验的区分度
- **Hallo4** 已经是 Hallo 系列第四代，审稿人可能会问"为什么基于 Hallo3 而非 Hallo4" → 需要说明 Hallo4 代码/权重是否公开，以及多语言方向的独立性
- **2026 趋势**：DiT 已成主流（READ、PC-Talk、OmniSync），多语言是新兴热点（Polyglot、AudioFace），我们的方向**没有过时但竞争在加剧**，需要尽快出实验结果

