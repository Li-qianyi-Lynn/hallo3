# 2026-10-08：训练 Loss 分析

承接 [2026-10-07_queue_devel_syncnet.md](2026-10-07_queue_devel_syncnet.md)。

## 一、当前进度

训练累计约 **10000 / 13600** 步，决定不跑满 13600，原因：

- 数据集只有 2702 条 clip（~3.8 小时），13600 步 ≈ 10 epoch，小数据集有过拟合风险
- Val 集选 checkpoint 机制可以保证取到最好的结果，不依赖跑满步数
- 已有 val / eval 集，可以直接评估

---

## 二、Loss 计算方式

框架为 **DDPM v-prediction**（LTX-2 / CogVideoX 风格），实现在 `hallo3/sgm/modules/diffusionmodules/loss.py`。

### 步骤

**① 加噪（forward process）**

```
z_t = √ᾱₜ · x₀  +  √(1−ᾱₜ) · ε
```

- `x₀`：LTX VAE 编码后的干净视频 latent
- `ε ~ N(0, I)`：标准高斯噪声
- `√ᾱₜ`（`alphas_cumprod_sqrt`）：从 noise schedule 中随机采样，决定当前加噪程度

**② 模型预测**

网络输入 `z_t`，经 EDM 缩放（`c_skip, c_out, c_in`）后，输出**预测的干净 latent x̂₀**（v-prediction 等价形式）。

条件信息通过 cross-attention 注入：
- 音频条件：LTX-2 Audio VAE 编码后的 128 维特征（`AudioProjModel` 映射到模型维度）
- 参考帧：ReferenceNet 提取特征后写入各层 attention bank

**③ 加权 MSE（loss）**

```
L = E_t [ wₜ · ‖ x̂₀ − x₀ ‖² ]

wₜ = 1 / (1 − ᾱₜ)    （SNR 加权，高信噪比 timestep 权重更大）
```

日志里打印的 `total loss` 即为 batch 内各样本该值的均值，单位是 latent 空间 MSE。

---

## 三、Loss 曲线观察

用 `scripts/plot_loss.py` 合并 22 个 log 文件，共解析 **12981 个数据点**。

| 指标 | 数值 |
|------|------|
| Loss 最高 | 4.41e-01（session 重启 warmup 尖峰） |
| Loss 最低 | 3.39e-02 |
| 稳定区间 | **0.05–0.09**（step ~2500 之后） |

### 现象

- Loss 从初始 ~0.11 快速下降到 ~0.07–0.09，之后全程维持平稳，**没有明显的持续下降趋势**
- 曲线中有多处高尖峰（最高 ~0.44），对应每次手动重启 training session 时 **lr warmup 重置**，跑几十步后自动恢复正常
- Smoothed 曲线从头到尾基本水平

### 结论

**Loss 平对扩散模型是正常现象**，原因：

1. 训练 loss 是所有 noise timestep 的加权平均 MSE（v-prediction），模型在早期就能收敛到一个基线水平
2. 后续生成质量的提升（口型精度、细节）体现在高信噪比 timestep 上，在整体平均 loss 里变化很小
3. 真正的效果需要生成视频后用 SyncNet / 人工评估来判断，loss 曲线参考价值有限

---

## 三、数据清理（本日完成）

- 用 `scripts/find_silent_audio.py` 扫描 T7 全量 .m4a（共 215,835 个），发现 **451 个静音文件**（max_volume < −60 dB）
- 从 T7 删除对应 .m4a 和 .mp4，共删除 **920 个文件**
- 生成清理后的数据集文件：
  - `talkvid_datafiles/filtered_video_clips_no_silent.json`（188,112 条）
  - `talkvid_datafiles/filtered_video_clips_with_captions_no_silent.json`（181,750 条）
- 当前训练用的 4k / 8k split 不受影响（扫描发现 0 个命中）

---

## 四、下一步

1. **选 checkpoint**：在 val 集上跑 SyncNet，选 confidence score 最高的 checkpoint（候选：step ~6500–10000 区间）
2. **生成 val 视频**：取 5–10 条 val 视频用选出的 checkpoint 跑 `sample_video.py`，肉眼对比口型
3. **eval 集评估**：确定 checkpoint 后，在 350 条 eval 上算 LSE-D / LSE-C，对比 pretrained Hallo3
4. **补 A 组对照**：用原 wav2vec 编码器在同一 2702 条数据上训同样步数
