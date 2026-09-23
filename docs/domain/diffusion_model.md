# Diffusion Model（扩散模型）通俗解释

---

## 一句话

**给图片/视频不断加噪声变成乱码，然后训练模型学会怎么一步步去掉噪声还原回来。**

---

## 比喻

```
想象你有一张清晰的照片：

第 0 步:  清晰人脸
第 1 步:  人脸 + 少量噪点
第 2 步:  人脸 + 更多噪点
  ...
第 999步: 纯噪声（完全看不出是什么）

这个"加噪声"的过程叫 Forward Diffusion（前向扩散）
→ 这一步不需要学习，纯数学公式，每次加一点随机噪声


训练时，让模型反过来学：

第 999步: 纯噪声
第 998步: 稍微有点轮廓了     ← 模型预测"这里应该去掉多少噪声"
  ...
第 1 步:  快清楚了
第 0 步:  清晰人脸还原!

这个"去噪声"的过程叫 Reverse Diffusion（逆向扩散）
→ 这一步需要训练模型来学
```

---

## 训练具体怎么做

就是 Hallo3 每个 iteration 在做的事：

```python
1. 拿一段真实视频                    ← batch["mp4"]
2. 随机选一个噪声等级（比如第 500 步） ← sigma_sampler
3. 加对应强度的噪声                   ← noised_input = video + noise
4. 把 [加噪视频 + 音频 + 人脸 + 文本] 喂给模型
5. 模型预测: "我觉得加的噪声长这样"    ← model(noised_input, audio_emb, ...)
6. loss = |预测噪声 - 真实噪声|²      ← MSE loss
7. backward, 调参数，让预测更准
```

对应到代码：

| 步骤 | 代码位置 |
|------|---------|
| 1. 取视频 | `diffusion_video.py:198` shared_step → get_input |
| 2-3. 加噪声 | `loss.py:76` VideoDiffusionLoss 内部 |
| 4-5. 模型预测噪声 | `denoiser.py:39` → `dit_video_concat.py:842` → 42层 Transformer |
| 6. 算 loss | `loss.py` MSE(预测噪声, 真实噪声) |
| 7. 反向传播 | PyTorch/DeepSpeed 自动完成 |

---

## 推理（生成视频）时

```
1. 从纯随机噪声开始                   ← torch.randn(...)
2. 重复 50 次:
     模型看着 [噪声 + 音频 + 人脸 + 文本]
     预测噪声 → 减掉一点噪声 → 图像稍微清晰一点
3. 50 步后: 噪声变成了说话的视频
```

对应到代码：`diffusion_video.py:292` sample() 方法，调用 sampler 做 50 步去噪。

---

## 为什么叫 "扩散"

名字来自物理学——一滴墨水滴进水里会慢慢扩散开（加噪声 = 信息扩散消失）。模型学的是**把扩散的墨水收回来**（去噪声 = 信息恢复）。

---

## ANN（人工神经网络）全景图

### 什么是 ANN

ANN (Artificial Neural Network) = 人工神经网络，是所有深度学习的基础。

**核心思想**：模仿人脑神经元的连接方式，用大量简单的数学运算（矩阵乘法 + 激活函数）堆叠起来，让机器学会从数据中找规律。

```
一个"神经元"做的事：

输入 x1, x2, x3
  ↓
加权求和: y = w1·x1 + w2·x2 + w3·x3 + b
  ↓
激活函数: output = ReLU(y)    ← 如果 y>0 输出 y，否则输出 0
  ↓
输出

成千上万个这样的神经元连在一起 = 神经网络
训练 = 调整所有的 w（权重）让输出越来越准
```

### ANN 的分类

```
ANN (人工神经网络) ← 最大的范畴，所有深度学习都是 ANN
│
├── 按网络结构分
│   │
│   ├── MLP (多层感知机 / 全连接网络)
│   │   最基础的形式，每层每个神经元连接下一层所有神经元
│   │   Hallo3 中: AudioProjModel 的 proj1/proj2/proj3
│   │
│   ├── CNN (卷积神经网络)
│   │   用滑动窗口（卷积核）扫描数据，擅长处理图像/音频的局部特征
│   │   Hallo3 中: AudioEncoder (2D卷积压缩mel频谱)
│   │             3D-VAE (3D卷积压缩视频)
│   │             Conv1d (1D卷积压缩时间维度)
│   │
│   ├── RNN / LSTM (循环神经网络)
│   │   按时间顺序处理序列数据，有"记忆"
│   │   Hallo3 中: 没有用（已被 Transformer 取代）
│   │
│   └── Transformer (注意力机制网络)
│       用 Attention 机制让每个位置都能看到所有其他位置
│       Hallo3 中: DiffusionTransformer 42层 (核心模型)
│                  T5-xxl (文本编码)
│                  Wav2Vec2 (旧音频编码)
│
├── 按任务类型分
│   │
│   ├── 判别模型 (Discriminative) ← "这是什么？"
│   │   ├── 图像分类: 这张图是猫还是狗？
│   │   ├── 语音识别: 这段音频说了什么？(Wav2Vec2 原本干的事)
│   │   └── 目标检测: 图里的人脸在哪？
│   │
│   └── 生成模型 (Generative) ← "造一个出来！" ← Hallo3 属于这类
│       ├── GAN (对抗生成网络)          ← 2014, 两个网络对打
│       ├── VAE (变分自编码器)          ← 2013, 编码-解码
│       ├── Diffusion Model (扩散模型)  ← 2020 起爆发 ← Hallo3 核心
│       └── Flow Matching (流匹配)      ← 2023, 扩散模型的改进版
│
└── 按训练方式分
    ├── 监督学习 (Supervised)     ← 有标签，如分类
    ├── 无监督学习 (Unsupervised) ← 无标签，如聚类
    ├── 自监督学习 (Self-supervised) ← 数据本身当标签
    │   Wav2Vec2 就是自监督：用语音自己预测自己
    └── Hallo3 的训练方式:
        给视频加噪声 → 让模型去噪 → 和原视频对比
        严格来说是 self-supervised（真实视频既是输入也是目标）
```

### Hallo3 用到了哪些

```
Hallo3 里的网络类型:

┌───────────────────────────────────────────────────────┐
│  ANN (人工神经网络)                                     │
│                                                        │
│  ├─ CNN (卷积神经网络)                                  │
│  │   ├─ AudioEncoder         ← 音频压缩，3级卷积下采样   │
│  │   ├─ 3D-VAE               ← 视频压缩                │
│  │   └─ Conv1d               ← AudioProjModel 时间压缩  │
│  │                                                     │
│  ├─ Transformer (注意力机制网络)                         │
│  │   ├─ DiffusionTransformer (42层) ← 核心，生成视频    │
│  │   ├─ T5-xxl               ← 文本编码                │
│  │   └─ Wav2Vec2 (旧链路)     ← 音频编码               │
│  │                                                     │
│  └─ MLP (全连接网络)                                    │
│      ├─ AudioProjModel proj1/2/3  ← 音频投影            │
│      └─ FaceProjModel             ← 人脸投影            │
└───────────────────────────────────────────────────────┘
```

---

## 扩散模型在生成模型中的位置

```
生成模型 (Generative Models) ← "造一个出来"
│
├── GAN (对抗生成网络)          ← 2014, 两个网络对打
│   生成器造假图，判别器判真假，互相博弈
│   优点: 生成速度快
│   缺点: 训练不稳定，容易"模式崩塌"(只会生成几种图)
│
├── VAE (变分自编码器)          ← 2013, 编码-解码
│   把数据压缩成 latent → 从 latent 重建回来
│   Hallo3 中: 3D-VAE 压缩视频, Audio VAE 压缩音频
│   优点: 训练稳定
│   缺点: 生成的图像偏模糊
│
├── Diffusion Model (扩散模型)  ← 2020 起开始爆发 ← Hallo3 核心
│   加噪声 → 训练模型去噪声
│   ├── DDPM                   ← 最早的扩散模型论文
│   ├── Stable Diffusion       ← 文本→图片（Stability AI）
│   ├── DALL-E                 ← 文本→图片（OpenAI）
│   ├── Sora                   ← 文本→视频（OpenAI）
│   ├── CogVideoX              ← 文本→视频（智谱 AI）
│   └── Hallo3                 ← 音频+人脸→说话视频 ← 你在做的
│   优点: 生成质量最高，训练稳定
│   缺点: 生成速度慢（要跑 50 步去噪）
│
└── Flow Matching (流匹配)      ← 2023, 扩散模型的改进版
    把加噪/去噪简化为一条直线路径
    优点: 比扩散模型更快
    缺点: 比较新，生态还不成熟
```

---

## 和 Hallo3 的关系

```
普通扩散模型:    噪声 + 文本描述           → 去噪 → 生成图片/视频
Hallo3 扩散模型: 噪声 + 文本 + 音频 + 人脸 → 去噪 → 生成说话视频
                                ↑
                       你做的工作: 改进音频条件的输入方式
                       Wav2Vec2 (英语) → LTX-2 VAE (语言无关)
```

**扩散模型是生成视频的引擎，音频编码器是告诉引擎"嘴巴该怎么动"的指令。** 你改的是指令的编码方式，引擎本身没动。

---

## 关键术语英文对照

| 中文 | 英文 |
|------|------|
| 扩散模型 | Diffusion Model |
| 前向扩散（加噪） | Forward Diffusion / Forward Process |
| 逆向扩散（去噪） | Reverse Diffusion / Reverse Process |
| 去噪器 | Denoiser |
| 噪声调度 | Noise Schedule |
| 采样步数 | Sampling Steps (Hallo3 用 50 步) |
| 条件生成 | Conditional Generation |
| 无分类器引导 | Classifier-Free Guidance (CFG) |
| 扩散 Transformer | Diffusion Transformer (DiT) |
| 潜空间扩散 | Latent Diffusion (在压缩后的 latent 上做扩散，不在原始像素上) |
