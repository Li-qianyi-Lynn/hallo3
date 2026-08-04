# 知识点整理 — 面试复习

本文档记录项目中涉及的机器学习/深度学习知识点，用于面试准备。

---

## 一、训练基础概念

### Iteration vs Epoch

- **1 iteration** = 模型用 1 个 batch 的数据做一次前向 + 反向传播 + 参数更新
- **1 epoch** = 把整个训练集完整看一遍
- **关系：** `1 epoch = 数据集大小 / batch_size` 个 iterations

**本项目示例：**
```
数据集：59,962 条视频
batch_size：1（单卡）
→ 1 epoch = 59,962 iterations
→ 100 iterations = 0.17% 的数据，远不到 1 epoch
```

**面试要点：** iteration 和 epoch 的区别，以及 batch_size 对训练速度和稳定性的影响。

---

### Batch Size 对训练的影响

| batch_size | 梯度噪声 | 训练稳定性 | 显存占用 | 收敛速度 |
|------------|---------|-----------|---------|---------|
| 小（1~4） | 大 | 不稳定，loss 震荡 | 低 | 慢但泛化好 |
| 大（64+） | 小 | 稳定，loss 平滑 | 高 | 快但可能过拟合 |

**本项目：** batch_size=1，loss 在 0.07~0.72 间剧烈震荡，属正常现象。

### Batch Size 详解

**1. 训练速度**

batch_size 越大，GPU 利用率越高，但不是线性加速：
```
batch_size 1  → GPU 大量核心空闲
batch_size 8  → 不是 8x 快，可能只有 4~6x（内存带宽成瓶颈）
batch_size 64 → 边际收益递减
```

**本项目：** 单卡 batch_size=1，GPU 利用率偏低，每个 iteration 约 7,569ms。4 卡时 effective batch_size=4，理论加速约 3~4x（非线性）。

---

**2. 梯度噪声与 loss 稳定性**

- **小 batch（=1）：** 每次只看 1 个样本，梯度估计不准，loss 震荡大
- **大 batch：** 梯度是多样本的平均，更接近真实方向，loss 曲线平滑

**本项目实测：** 100 iterations，batch_size=1，loss 在 0.07~0.72 大幅震荡，最高 spike 0.716。如果 batch_size=8，同样 100 iterations 的 loss 曲线会平滑得多。

---

**3. 泛化性能（面试常考）**

大 batch 反而可能泛化更差：
- 大 batch 梯度准，容易陷入 **sharp minima**（尖锐局部最优）
- 小 batch 的随机噪声帮助跳出尖锐最优，更容易找到 **flat minima**（平坦最优）
- Flat minima 对测试集泛化更好

```
Loss 曲面示意：
sharp minima      flat minima
     /\               ___
    /  \           __/   \__
───/    \───    ──/         \──
 小扰动就跳出    鲁棒，泛化好
```

**本项目：** batch_size=1 的噪声大，对只有 6 万样本的 fine-tuning 来说反而有一定正则化作用，不容易过拟合。

---

**4. 学习率需要配合调整（Linear Scaling Rule）**

```
batch_size 扩大 k 倍 → 学习率也建议扩大 k 倍
```

**本项目：** 当前 lr=1e-5，batch_size=1。如果改为 4 卡（effective batch=4），理论上 lr 应调整为 4e-5。

---

**面试一句话总结：**
> 大 batch 训练更快更稳定，但需要更大学习率，且可能牺牲泛化性（sharp minima）；小 batch 噪声大但有隐式正则化效果，适合资源受限或数据量有限的 fine-tuning 场景。

---

### 数据采样策略（DataLoader）

本项目 `data_video.py` 中使用了两层随机：

**第一层：视频选择（shuffle_buffer）**
- 使用 `shuffle_buffer=1000`：维护一个 1000 个样本的缓冲区，每次从中随机抽取
- 不是严格随机全局 shuffle，而是局部随机（流式数据集的常用方案）
- 优点：内存占用低，适合大数据集

**第二层：视频片段选择（随机起始帧）**
```python
start = random.randint(0, ori_vlen - sample_len - 1)
```
- 每次训练从视频中随机选起始帧，取连续 49 帧
- 同一个视频每次 epoch 看到的片段不同 → **数据增强**
- 有效增加了数据多样性

**面试要点：** 为什么用 shuffle_buffer 而不是全局 shuffle？流式数据集的权衡（内存 vs 随机性）。

---

## 二、多卡训练

### NCCL（NVIDIA Collective Communications Library）

- **作用：** 多 GPU 训练时负责 GPU 之间的梯度同步（all-reduce）
- **原理：** 每张卡算完各自的梯度后，通过 NCCL 做 all-reduce（所有梯度求平均），再各自更新参数

```
GPU0 ──┐
GPU1 ──┤  NCCL all-reduce  ──→  每卡得到平均梯度 → 各自更新参数
GPU2 ──┤
GPU3 ──┘
```

- **为什么比 gloo 快：** 走 NVLink/PCIe GPU 直连，不经 CPU；gloo 走 CPU 内存，慢 10~100 倍
- **单卡为什么不需要：** world_size=1，无需卡间通信，用 gloo 完全等价

**本项目遇到的问题：** NCCL 2.29.7 + CUDA 12.8 驱动不兼容，单卡用 `--distributed_backend gloo` 绕过。

---

### DeepSpeed ZeRO（Zero Redundancy Optimizer）

| ZeRO Stage | 分片内容 | 显存节省 | 通信开销 |
|------------|---------|---------|---------|
| Stage 1 | optimizer states | ~4× | 低 |
| Stage 2 | + gradients | ~8× | 中 |
| Stage 3 | + model params | ~64× | 高 |

**本项目：** 使用 ZeRO-2，在节省显存的同时保持合理通信开销。

**面试要点：** ZeRO 解决了什么问题（模型太大单卡放不下），三个 stage 各分片什么内容。

---

## 三、扩散模型训练

### 扩散模型的 Loss

扩散模型训练的 loss 是**噪声预测误差**：
```
loss = MSE(预测的噪声, 实际加的噪声)
```
- 不直接优化生成质量，而是学习如何预测噪声
- Loss 下降 ≠ 生成质量直接提升（但相关）
- **本项目 loss 范围：** 0.07~0.15（正常），偶发 spike 来自困难样本

**面试要点：** 为什么扩散模型用噪声预测而不是直接重建？（数学稳定性，score matching）

---

### Fine-tuning vs 从头训练

| 方式 | 数据需求 | 计算需求 | 适用场景 |
|------|---------|---------|---------|
| 从头训练 | 海量（TB级） | 极大 | 构建基础模型 |
| Full fine-tuning | 中等（GB级） | 大 | 领域适配 |
| LoRA/部分参数 | 少（GB级以下） | 小 | 快速适配 |

**本项目：** Fine-tuning 只训练 `attention` + `face` 模块（`not_trainable_prefixes: ['audio']`），冻结其他参数。

---

## 四、显存优化技术

### Activation Checkpointing（梯度检查点）

- **问题：** 反向传播需要保存所有中间激活值，显存占用巨大
- **方案：** 前向时不保存激活值，反向时重新计算
- **权衡：** 显存减少 ~50%，计算时间增加 ~30%
- **本项目：** `cpu_checkpointing: true`，激活值 offload 到 CPU

### xformers（Memory-Efficient Attention）

- **标准 attention：** O(n²) 显存，n=序列长度
- **xformers：** O(n) 显存，使用 FlashAttention 算法
- **本项目效果：** 17,550 tokens 的序列，每层标准 attention 需 ~29GB，xformers 大幅降低
- **必须安装：** `pip install xformers==0.0.28.post1 --index-url https://download.pytorch.org/whl/cu124`

---

## 五、本项目训练规模参考

| 配置 | batch_size | 1 epoch iterations | 30k iter 用时 |
|------|------------|-------------------|--------------|
| 1 × H200 | 1 | 59,962 | ~5 天 |
| 4 × H200 | 4 | 14,991 | ~30 小时 |
| 8 × H200 | 8 | 7,496 | ~15 小时 |
| 8 × A100 | 8 | 7,496 | ~10 小时 |

**Fine-tuning 建议 iterations：** 30,000~60,000（约 0.5~1 epoch）
