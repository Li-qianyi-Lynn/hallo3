# Research Plan

## 一、短期（本周内）—— 验证 Path 1 的假设

### 1. 数据侧：确认 English vs. Non-English 对比设置

- 从 TalkVid Core（254h，9种语言）里划分出英语子集和非英语子集
- 明确训练/测试划分，确保有 ground-truth video 可用于评估（这是导师强调的硬性要求——没有对应 ground-truth 的跨语言样本不能用于自动指标计算）

### 2. 模型侧：着手替换音频编码器（核心转折点，优先级最高）

- 调研 LTX 2.5 / RTX 2.5 的 audio encoder 实现，确认能否从代码库里独立抽出并适配到 Hallo3
- 对比其 transformer 架构与 Hallo3 现有 wav2vec-MLP 的输入输出维度，评估适配成本
- 先在已经跑通的小规模 pipeline（100 iterations / 42 videos）上做替换测试，验证 forward pass 是否跑通

## 二、中期（2-3周）—— Ablation 设计

### 1. 设计消融实验矩阵

拆分两个变量的贡献：

| # | 编码器 | 数据 | 说明 |
|---|--------|------|------|
| A | 原编码器 | 英语为主 | Baseline |
| B | 新编码器 | 英语为主 | 隔离编码器的贡献 |
| C | 原编码器 | 平衡数据 | 隔离数据的贡献 |
| D | 新编码器 | 平衡数据 | 完整方案 |

### 2. 建立评估协议

- **自动指标**：仅在有 ground-truth 的语言对上计算（如 lip-sync score, LSE-D/LSE-C 等）
- **人工评估**：设计一个简单的 visual quality 打分表，尤其针对 zero-shot 的小语种样本
- **自比较（self-comparison）** 消融的具体形式要提前定义好，避免后期返工


## 三、优先级建议

> **音频编码器替换 > 消融实验设计 > 完整数据集微调（30K iterations）**
>
> 导师明确指出这是提升项目新颖性和效果的关键杠杆，而不仅仅是数据平衡问题。
