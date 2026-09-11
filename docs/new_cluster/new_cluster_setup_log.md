# 新集群环境搭建 & LTX-2 Audio VAE 验证日志

集群：AICR  
日期：2026-09-10  
GPU：RTX PRO 6000 Blackwell Server Edition (sm_120)

---

## 1. 环境搭建

### 1.1 Conda 环境创建

```bash
module load conda/latest
conda create -p /scratch/li_qiany_neu/envs/hallo3 python=3.10 -y
conda activate /scratch/li_qiany_neu/envs/hallo3
```

### 1.2 PyTorch 安装 — Blackwell GPU 需要 cu128

**问题**：最初装的 `torch==2.4.0+cu121` 不支持 Blackwell GPU (sm_120)，报错：
```
NVIDIA RTX PRO 6000 Blackwell Server Edition with CUDA capability sm_120 
is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
```

**解决**：升级到 torch 2.7.1 + cu128：
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

### 1.3 setuptools 版本冲突

**问题**：`pytorch_lightning` 依赖 `pkg_resources`，但新版 setuptools (84.0.0) 移除了它：
```
ModuleNotFoundError: No module named 'pkg_resources'
```

**解决**：降级 setuptools：
```bash
pip install setuptools==70.0.0
```

### 1.4 pyav 包名错误

**问题**：`requirements.txt` 中的 `pyav==14.0.1` 在 PyPI 上找不到：
```
ERROR: No matching distribution found for pyav==14.0.1
```

**解决**：这个包在 PyPI 上叫 `av`，不叫 `pyav`。从 requirements.txt 删除 pyav 行，单独装：
```bash
sed -i '/pyav/d' requirements.txt
pip install av
pip install -r requirements.txt
```

### 1.5 deepspeed 安装需要 CUDA

**问题**：登录节点没有 CUDA runtime，deepspeed 构建失败：
```
FileNotFoundError: [Errno 2] No such file or directory: '/usr/local/cuda/bin/nvcc'
```

**解决**：先加载 CUDA 模块再装：
```bash
module load cuda
pip install -r requirements.txt
```

### 1.6 torchaudio 版本不匹配

**问题**：升级 torch 到 2.7.1 后，torchaudio 还是旧版本，ABI 不兼容：
```
OSError: undefined symbol: torch_library_impl
```

**解决**：torchaudio 版本必须和 torch 完全匹配：
```bash
pip install torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

---

## 2. 集群使用

### 2.1 查看账户名

```bash
sacctmgr show user $USER withassoc format=user,account%30
# 结果：p2026_0014_neu
```

### 2.2 申请交互式 GPU 节点

```bash
salloc --partition=rtx-devel --gpus=1 --cpus-per-task=8 --mem=32G --time=00:30:00 --account=p2026_0014_neu
```

可用分区：
| 分区 | GPU | 最大时间 |
|------|-----|----------|
| `rtx-devel` | RTX PRO 6000 | 4h |
| `rtx-batch` | RTX PRO 6000 | 24h |
| `b200-devel` | B200 | 4h |
| `b200-batch` | B200 | 24h |

### 2.3 进入节点后的环境加载

```bash
module load miniforge3
module load cuda
conda activate hallo3  # 或 conda activate /scratch/li_qiany_neu/envs/hallo3
```

---

## 3. LTX-2 Audio VAE 验证

### 3.1 下载权重

需要先在 https://huggingface.co/Lightricks/LTX-2.5 申请访问权限，然后：
```bash
huggingface-cli login  # 粘贴 token
huggingface-cli download Lightricks/LTX-2.5 \
    vae/ltx-2.5-audio-vae-bf16.safetensors \
    --local-dir /scratch/li_qiany_neu/pretrained_models/ltx2.5
```

### 3.2 AudioEncoder 实例化参数 — mid_block_add_attention

**问题**：用默认参数 `mid_block_add_attention=True` 实例化 AudioEncoder，导致 8 个 attention key missing：
```
Missing keys: ['mid.attn_1.q.weight', 'mid.attn_1.q.bias', 
'mid.attn_1.k.weight', 'mid.attn_1.k.bias', ...]
```

**原因**：checkpoint 的 mid block 没有 attention 层，说明训练时 `mid_block_add_attention=False`。

**解决**：实例化时设 `mid_block_add_attention=False`：
```python
encoder = AudioEncoder(
    ch=128,
    ch_mult=(1, 2, 4),
    num_res_blocks=2,
    attn_resolutions={8, 16, 32},
    resolution=256,
    z_channels=8,
    double_z=True,
    in_channels=2,
    norm_type=NormType.PIXEL,
    causality_axis=CausalityAxis.HEIGHT,
    sample_rate=16000,
    mel_hop_length=160,
    n_fft=1024,
    is_causal=True,
    mel_bins=64,
    mid_block_add_attention=False,  # checkpoint 中无 attention 权重
)
```

### 3.3 enum 参数不能传字符串

**问题**：`norm_type="pixel"` 和 `causality_axis="height"` 传字符串导致 match/case 不匹配：
```
ValueError: Invalid causality_axis: height
```

**解决**：必须传枚举值：
```python
from hallo3.sgm.models.ltx_audio_vae.normalization import NormType
from hallo3.sgm.models.ltx_audio_vae.causality_axis import CausalityAxis

norm_type=NormType.PIXEL,
causality_axis=CausalityAxis.HEIGHT,
```

### 3.4 验证通过的结果

```
  Encoder loaded:       OK (46/46 keys matched)
  Forward pass:         OK
  Output shape:         (1, 8, 251, 16) → patchified (1, 251, 128)
  Temporal fps:         25.1 (target: 25.0) ✓
  AudioProjModel input: 640 (was 46080, ratio: 72.0x)
```

---

## 4. 正确的 AudioEncoder 配置参数总结

```python
AudioEncoder(
    ch=128,                          # base channels
    ch_mult=(1, 2, 4),               # 3 levels, 2 downsamples → 4x
    num_res_blocks=2,                # ResNet blocks per level
    attn_resolutions={8, 16, 32},    # (实际未触发，resolution 从 256 降到 64)
    resolution=256,
    z_channels=8,                    # latent channels
    double_z=True,                   # output 16ch, take first 8 as mean
    dropout=0.0,
    resamp_with_conv=True,
    in_channels=2,                   # stereo
    norm_type=NormType.PIXEL,        # PixelNorm (不是 GroupNorm)
    causality_axis=CausalityAxis.HEIGHT,
    sample_rate=16000,
    mel_hop_length=160,
    n_fft=1024,
    is_causal=True,
    mel_bins=64,
    mid_block_add_attention=False,   # 此 checkpoint 无 mid attention
)
```
