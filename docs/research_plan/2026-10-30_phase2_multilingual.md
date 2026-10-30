# 阶段二：多语言消融实验计划

**前置条件**：阶段一 G1（Omnilingual + adapter，英语 2700 条）训练完成且 conf > 3，确认 encoder 替换方案可行。

---

## 一、研究问题

在 G1 验证 Omnilingual encoder 在英语上恢复口型同步后，进一步回答：

1. **多语言训练数据**是否能提升非英语的口型同步质量？
2. **Omnilingual encoder vs 原版 wav2vec2-base**，在多语言数据上差距有多大？
3. 编码器和数据各自的贡献能否通过 2×2 消融矩阵拆分？

---

## 二、可用数据盘点

### 2.1 非英语数据（`splits/talkvid_non_english.json`）

| 语言 | clips | 占比 | 备注 |
|------|-------|------|------|
| Chinese | 22,434 | 86.0% | 主力 |
| Spanish | 2,213 | 8.5% | |
| Hindi | 955 | 3.7% | |
| Portuguese | 170 | 0.7% | 太少 |
| French | 130 | 0.5% | 太少 |
| Other | 162 | 0.6% | 太少 |
| Arabic | 33 | 0.1% | 太少 |
| **合计** | **26,097** | | |

另有 Unknown 语言 7,432 条（可能混杂英语和其他语言，暂不使用）。

### 2.2 现有评估集

| 集合 | 数量 | 语言 | 问题 |
|------|------|------|------|
| val5 | 5 | 全英语 | 不能评估多语言 |
| val200 | 200 | 全英语 | 不能评估多语言 |

**关键缺口**：没有非英语评估集，阶段二必须先构建。

---

## 三、评估集构建

### 3.1 多语言评估集 eval_multilingual

从 `talkvid_non_english.json` 中按语言分层抽样，构建多语言评估集：

| 语言 | 抽取数量 | 说明 |
|------|----------|------|
| Chinese | 100 | 主力语言，充足 |
| Spanish | 50 | 次多语言 |
| Hindi | 50 | 第三语言 |
| **合计** | **200** | 与 val200 规模对齐 |

French/Portuguese/Arabic/Other 数据太少（<200 条），不纳入正式评估，全部留作训练。

抽样脚本（待写）：
```bash
python scripts/make_multilingual_eval.py \
    --input splits/talkvid_non_english.json \
    --output splits/talkvid_eval_multilingual.json \
    --chinese 100 --spanish 50 --hindi 50 \
    --seed 42
```

### 3.2 评估方式

- **SyncNet**：同阶段一，Confidence / Min dist / AV offset
- **按语言分组报告**：每种语言单独统计指标，不混在一起平均
- **Ground Truth 上限**：多语言评估集也要先跑 B0（真值视频 SyncNet 分数），作为各语言的上限参照

---

## 四、训练数据构建

### 4.1 数据策略

目标：构建一份**语言平衡**的训练集，与阶段一的 2700 条英语对等。

**方案：英语 + 多语言混合，总量保持 2700 条**

| 语言 | 数量 | 来源 | 说明 |
|------|------|------|------|
| English | 1350 | 从阶段一的 2700 条中随机抽 50% | 保留英语能力 |
| Chinese | 900 | 从非英语集剔除 eval 后抽取 | 最大非英语语种 |
| Spanish | 300 | 同上 | |
| Hindi | 150 | 同上 | 尽量多取（总共 955，扣除 eval 50 后剩 905） |
| **合计** | **2700** | | 与阶段一规模对等，控制数据量变量 |

> 为什么不用全部 26K 非英语？为了和阶段一控制变量（总量 2700，步数 7000），隔离"数据语言分布"这一个因素。

生成脚本（待写）：
```bash
python scripts/make_multilingual_train.py \
    --en-pool data/talkvid_train_2700.json \
    --non-en-pool splits/talkvid_non_english.json \
    --eval-exclude splits/talkvid_eval_multilingual.json \
    --english 1350 --chinese 900 --spanish 300 --hindi 150 \
    --seed 42 \
    --output data/talkvid_train_2700_multilingual.json
```

### 4.2 数据预处理

非英语数据需要和阶段一相同的预处理流程：

```bash
# Step 1: 提取 images, face_emb, face_mask
python ../scripts/process_4k_data.py \
    --json ../data/talkvid_train_2700_multilingual.json \
    --data-dir /scratch/li_qiany_neu/talkvid_8k \
    --skip_steps audio_emb

# Step 2: 用 Omnilingual wav2vec2 提取 audio embedding
python ../scripts/re_extract_audio_emb.py \
    --json ../data/talkvid_train_2700_multilingual.json \
    --encoder /scratch/li_qiany_neu/pretrained_models/omnilingual-w2v2-300m \
    --output-dir /scratch/li_qiany_neu/talkvid_8k/audio_emb_omnilingual \
    --output-json ../data/talkvid_train_2700_multilingual_omnilingual.json
```

**注意**：非英语视频可能尚未下载到 `/scratch/li_qiany_neu/talkvid_8k/`，需要先确认数据是否存在，必要时从 TalkVid 源重新下载。

---

## 五、实验设计（2×2 消融矩阵）

### 5.1 实验组

```
                        英语数据 (2700 条)         多语言混合数据 (2700 条)
                        ──────────────────      ─────────────────────
wav2vec2-base (英语)     G3 (阶段一已设计)          M1 (新)
Omnilingual (多语言)     G1 (阶段一主实验) ⭐        M2 (新) ⭐⭐
```

| 实验 | Encoder | 数据 | Adapter | 说明 |
|------|---------|------|---------|------|
| G1 | Omnilingual | 英语 2700 | ✅ | 阶段一主实验（已有） |
| G3 | wav2vec2-base | 英语 2700 | 无 | 阶段一对照（已有） |
| M1 | wav2vec2-base | 多语言 2700 | 无 | 新：隔离数据的贡献 |
| M2 | Omnilingual | 多语言 2700 | ✅ | 新：完整方案 ⭐⭐ |

### 5.2 变量控制

| 变量 | G1 | G3 | M1 | M2 |
|------|-----|-----|-----|-----|
| Encoder | Omnilingual | wav2vec2-base | wav2vec2-base | Omnilingual |
| Encoder 语言覆盖 | 1600+ 语言 | 仅英语 | 仅英语 | 1600+ 语言 |
| 训练数据 | 英语 2700 | 英语 2700 | 多语言 2700 | 多语言 2700 |
| Adapter | Linear(1024→768) | 无 | 无 | Linear(1024→768) |
| 训练步数 | 7000 | 7000 | 7000 | 7000 |

### 5.3 关键对比

| 对比 | 隔离的变量 | 回答的问题 |
|------|----------|-----------|
| M2 vs G1 | 训练数据（多语言 vs 英语） | 多语言数据对非英语口型有多大提升？ |
| M2 vs M1 | Encoder（Omnilingual vs wav2vec2） | 在多语言数据上，encoder 差距有多大？ |
| M1 vs G3 | 训练数据（多语言 vs 英语） | 仅换数据（不换 encoder），非英语能提升多少？ |
| M2 vs G3 | Encoder + 数据（完整方案 vs baseline） | 完整方案相比 baseline 的总提升 |

### 5.4 评估矩阵

每个实验组都在**两套评估集**上打分：

|  | val200（英语） | eval_multilingual（中/西/印） |
|--|-------------|-------------------------|
| G1 | ✅ | ✅ |
| G3 | ✅ | ✅ |
| M1 | ✅ | ✅ |
| M2 | ✅ | ✅ |

> 英语评估集检查"换了多语言数据后英语有没有退化"；多语言评估集检查"非英语有没有提升"。

---

## 六、预期结果与假设

### 6.1 英语评估集（val200）

| 预期排序 | 说明 |
|---------|------|
| G1 ≈ M2 > G3 ≈ M1 | Omnilingual encoder 在英语上应与 wav2vec2-base 持平或略好 |
| M2 不应显著低于 G1 | 混入多语言数据不应导致英语退化（如果退化，说明数据冲突） |

### 6.2 多语言评估集（eval_multilingual）

| 预期排序 | 说明 |
|---------|------|
| M2 > M1 > G1 > G3 | 完整方案最优；仅换数据也有提升；仅换 encoder 有一定提升；baseline 最差 |
| M2 >> G3 | 这是我们论文的核心结论——encoder + 数据共同作用 |

### 6.3 如果结果不符合预期

| 异常情况 | 可能原因 | 应对 |
|---------|---------|------|
| M2 ≈ G1（多语言数据没帮助） | 非英语数据质量差，或 2700 条中非英语占比太低 | 提高非英语占比（如 1:1:1:1），或增加总训练量 |
| M2 英语退化明显 | 英语数据从 2700 降到 1350 不够 | 调整比例（英语 1800 + 非英语 900） |
| M1 ≈ G3（wav2vec2-base 在非英语上也行） | wav2vec2-base 的英语音素对非英语也有一定迁移能力 | 这本身是一个有意义的发现，可以在论文中讨论 |

---

## 七、优先级与执行顺序

```
前置（阶段一完成后）
  0. 确认 G1 conf > 3，方案可行

阶段二第一轮：数据准备
  1. 确认非英语视频是否已在 scratch 上（ls 检查）
  2. 构建多语言评估集 eval_multilingual（200 条）
  3. 构建多语言训练集 talkvid_train_2700_multilingual（2700 条）
  4. 多语言评估集 B0 真值打分（各语言上限）
  5. 数据预处理（images, face_emb, face_mask, audio_emb）

阶段二第二轮：训练 + 评估
  6. M2 训练（Omnilingual + 多语言数据）— 主实验
  7. M2 推理 eval_multilingual + val200，SyncNet 打分
  8. G1 补充推理 eval_multilingual，SyncNet 打分（交叉对比）

阶段二第三轮：对照实验
  9. M1 训练（wav2vec2-base + 多语言数据）
  10. M1 推理 eval_multilingual + val200，SyncNet 打分
  11. G3 补充推理 eval_multilingual，SyncNet 打分

阶段二第四轮：结果分析
  12. 汇总 2×2 矩阵结果表
  13. 按语言分组分析（中文/西班牙语/印地语各自的提升幅度）
  14. 确认是否需要调整数据比例或增加训练步数
```

---

## 八、代码改动清单

| 文件 | 改动 | 状态 |
|------|------|------|
| `scripts/make_multilingual_eval.py` | 新增：从非英语集分层抽样构建评估集 | ⬜ 待写 |
| `scripts/make_multilingual_train.py` | 新增：按语言比例混合训练集 | ⬜ 待写 |
| 模型代码 | 无改动，复用阶段一 G1/G3 的代码 | — |
| 训练配置 | 仅改 `train_data` 指向新 JSON | ⬜ |
| SyncNet 评估脚本 | 需支持按语言分组统计 | ⬜ 待确认 |

---

## 九、风险评估

| 风险 | 级别 | 说明 | 缓解 |
|------|------|------|------|
| 非英语视频未在 scratch 上 | 高 | 8K quality 集全是英语，非英语数据可能需要重新下载 | 先 ls 确认，必要时从 TalkVid 源下载 |
| 非英语数据质量参差 | 中 | 没有经过 8K quality 级别的筛选 | 可以用 dover_scores 做质量过滤 |
| 中文占比过高（86%） | 中 | "多语言"实际上可能主要反映中文效果 | 评估时分语言报告，不混合平均 |
| scratch 30 天清理 | 低 | 已有 touch_scratch.sh 脚本 | 定期执行或设 cron job |
| 训练量不够 | 中 | 多语言数据分散在多个语言，每个语言的有效样本少 | 如果效果不好，考虑增加总量到 5000+ 或增加步数 |

---

## 十、成功标准

| 层级 | 标准 | 含义 |
|------|------|------|
| 最低 | M2 在 eval_multilingual 上 conf > M2 在 val200 的噪声基线 | 多语言训练至少在驱动口型 |
| 及格 | M2 在 eval_multilingual 上 conf > G1 在 eval_multilingual 上 conf | 多语言数据确实有帮助 |
| 良好 | M2 >> G3 在 eval_multilingual，且 M2 ≈ G1 在 val200 | 非英语大幅提升，英语不退化 |
| 优秀 | M2 在各语言的 eval_multilingual 上 conf 均 > 3，且英语不退化 | 完整方案在所有语言上基本同步 |

---

## 十一、与论文的对应

阶段二的 2×2 消融矩阵直接对应 `research_plan_new.md` 中设计的消融实验：

| research_plan_new 中的组 | 本计划对应 |
|------------------------|----------|
| A: 原编码器 + 英语为主 | G3 |
| B: 新编码器 + 英语为主 | G1 |
| C: 原编码器 + 平衡数据 | M1 |
| D: 新编码器 + 平衡数据 | M2 |

论文的核心 claim：**Omnilingual encoder + 多语言数据（M2）显著优于 wav2vec2-base + 英语数据（G3），且两个因素各有独立贡献（通过 G1 和 M1 拆分）。**
