# 问题排查记录

本文档记录 Hallo3 × TalkVid 项目中遇到的所有问题及解决方法。

---

## 一、环境配置问题

### 1. conda 环境目录已存在

**报错：**
```
CondaValueError: prefix already exists: /home/li.qianyi/envs/hallo
# 或
DirectoryNotACondaEnvironmentError
```

**原因：** 之前安装到一半失败，目录残留但不是合法 conda 环境。

**解决：**
```bash
rm -rf /home/li.qianyi/envs/hallo
conda create --prefix /home/li.qianyi/envs/hallo python=3.10 -y
```

---

### 2. pyav 安装失败

**报错：**
```
ERROR: Could not find a version that satisfies the requirement pyav==14.0.1
```

**原因：** `requirements.txt` 里的包名/版本有误，PyPI 上没有这个包。

**解决：**
```bash
grep -v "pyav==14.0.1" requirements.txt > requirements_fixed.txt
pip install -r requirements_fixed.txt
```

---

### 3. pkg_resources 缺失

**报错：**
```
ModuleNotFoundError: No module named 'pkg_resources'
```

**原因：** setuptools 版本过新，82+ 版本移除了 pkg_resources。

**解决：**
```bash
pip install "setuptools<82"
```

---

### 4. 登录节点加载 .pt 文件被 Kill

**报错：**
```
Killed
```

**原因：** 登录节点内存限制严格，加载大型 .pt 文件超限被系统杀死。

**解决：** 申请计算节点再操作：
```bash
srun -p sharing --gres=gpu:h200:1 --mem=60G --pty bash
```

---

## 二、训练问题

### 5. CUDA 无效设备（invalid device ordinal）

**报错：**
```
CUDA error: invalid device ordinal
```

**原因：** 训练脚本默认配置 8 卡，但实际只申请了 1 张 GPU，`CUDA_VISIBLE_DEVICES` 与 `nproc_per_node` 不匹配。

**解决：**
```bash
CUDA_VISIBLE_DEVICES="0" torchrun --standalone --nproc_per_node=1 ...
```

---

### 6. FusedEmaAdam CUDA 崩溃

**报错：**
```
CUDA error: illegal memory access
```

**原因：** SAT 框架自带的 `FusedEmaAdam` 是自定义 CUDA 算子，未针对 H200（Hopper 架构）编译。

**解决：** 在 `configs/sft_s1.yaml` 中修改 optimizer：
```yaml
# 改之前
optimizer:
  type: sat.ops.FusedEmaAdam

# 改之后
optimizer:
  type: AdamW
```

---

### 7. AdamW 不被 DeepSpeed 识别

**报错：**
```
torch.optim.AdamW is not supported DeepSpeed Optimizer
# 或
DeepSpeedCPUAdam is not supported
```

**原因：** DeepSpeed 只认特定的 optimizer 名称，不认带模块前缀的全路径，也不支持 `DeepSpeedCPUAdam`（需要额外编译）。

**解决：** 直接用 `AdamW`（不加任何前缀）：
```yaml
optimizer:
  type: AdamW
```

---

### 8. Slurm sharing 分区不支持 --time

**报错：**
```
srun: error: Requested time limit is invalid
```

**原因：** `sharing` 分区不允许指定时间限制。

**解决：** 去掉 `--time` 参数：
```bash
srun -p sharing --gres=gpu:h200:1 --mem=60G --pty bash
```

---

## 三、推理问题

### 9. 找不到 latest 元数据文件

**报错：**
```
ValueError: could not find the metadata file ./stage-1/train-stage-1-07-20-02-38/100/latest
```

**原因：** SAT 框架加载 checkpoint 的方式：先读父目录下的 `latest` 文件获取迭代号，再进对应子目录加载权重。直接指向 `100/` 子目录会失败。

**解决：**
```bash
# 1. 创建 latest 文件
echo "100" > stage-1/train-stage-1-07-20-02-38/latest

# 2. inference.yaml 的 load 路径改为父目录
# load: ./stage-1/train-stage-1-07-20-02-38   （不要加 /100）
```

---

## 四、TalkVid 数据下载问题

### 10. yt-dlp 未安装

**现象：** 提交 sbatch job 后立即失败，日志为空。

**解决：**
```bash
pip install yt-dlp
```

---

### 11. YouTube Bot 检测拦截

**报错：**
```
ERROR: Sign in to confirm you're not a bot. Use --cookies-from-browser or --cookies
```

**原因：** 集群 IP 被 YouTube 识别为爬虫，需要提供浏览器 cookies 证明身份。

**解决：**
```bash
# 本地 Mac 导出 cookies
yt-dlp --cookies-from-browser chrome --cookies ~/Desktop/youtube_cookies.txt --skip-download "https://www.youtube.com"

# 传到集群
rsync -avz ~/Desktop/youtube_cookies.txt "li.qianyi@login.explorer.northeastern.edu:/scratch/li.qianyi/TalkVid/youtube_cookies.txt"

# 提交时加上 --cookies 参数
python download_clips.py ... --cookies /scratch/li.qianyi/TalkVid/youtube_cookies.txt
```

---

### 12. 视频格式不可用

**报错：**
```
ERROR: Requested format is not available
```

**原因：** 下载脚本原本只允许 mp4 格式（`bestvideo[ext=mp4]+bestaudio[ext=m4a]`），但 YouTube 2023 年后大量视频只提供 webm（VP9 编码），没有 mp4 流。

**解决：** 修改 `download_clips.py` 放宽格式限制：
```python
# 改之前
"-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"

# 改之后
"-f", "bestvideo+bestaudio/best"
# 同时保留 --merge-output-format mp4 或 --remux-video mp4 自动转换
```

---

### 13. yt-dlp n-challenge 失败（未完全解决）

**报错：**
```
WARNING: n challenge solving failed
WARNING: Only images are available for download
```

**原因分析：**

| 层级 | 问题 |
|------|------|
| JS 运行时 | yt-dlp 检测不到 node，即使 `module load nodejs` 也无效（PATH 传递问题） |
| EJS solver | 安装了 `yt-dlp-ejs` 但未与 yt-dlp 联动 |
| PO Token | YouTube 新增 Proof of Origin Token，集群环境无法提供 |

**已尝试但无效的方法：**
- `--extractor-args "youtube:player_client=android"` — android 不支持 cookies，被跳过
- `--extractor-args "youtube:player_client=web"` — 触发 SABR streaming，格式列表为空
- `pip install -U yt-dlp`（升至 2026.07.04）— 版本最新，问题仍存在
- `module load nodejs/v22.11.0` — node 可用但 yt-dlp `JS runtimes: none`
- `pip install "yt-dlp[default]"`（安装 yt-dlp-ejs）— 仍然无效

**待尝试方向：**
- `--extractor-args "youtube:player_client=ios"`
- 刷新 cookies（当前 cookies 可能已过期）
- 排查 yt-dlp 为何检测不到已加载的 node

---

### 14. Python 日志在 sbatch 中不实时刷新

**现象：** `tail -f` 看日志长时间没有输出，不知道程序是否在运行。

**原因：** Python 在非交互模式（batch job）下默认缓冲 stdout，不实时写入文件。

**解决：** 在 sbatch 脚本里加：
```bash
export PYTHONUNBUFFERED=1
```

---

### 15. rsync 远程路径被 zsh 当本地路径解析

**报错：**
```
zsh: no such file or directory: li.qianyi@login.explorer.northeastern.edu:/path/...
```

**原因：** zsh 对包含特殊字符的路径尝试 glob 展开。

**解决：** 给远程路径加引号：
```bash
rsync -avz ~/local/file "li.qianyi@login.explorer.northeastern.edu:/remote/path/file"
```

---

### 16. yt-dlp 自动检测不到 conda 安装的 node（JS runtimes: none）

**现象：**
```
[debug] JS runtimes: none
[debug] [youtube] [jsc] JS Challenge Providers: bun (unavailable), deno (unavailable), node (unavailable), quickjs (unavailable)
WARNING: n challenge solving failed
WARNING: Only images are available for download
```

**原因分析（三层）：**

| 层级 | 问题 |
|------|------|
| 根本原因 | yt-dlp 2026.07.04 的 `_NodeJsRuntime` 自动检测逻辑无法识别 conda 安装的 node，即使 `shutil.which('node')` 能找到路径也无效 |
| 表现 | `_NodeJsRuntime(path=None).info` 返回 `None`，导致 node 被标记为不可用 |
| 结果 | 无 JS 运行时 → n-challenge 无法解 → 视频流 URL 无法解密 → 只能拿到缩略图 |

**诊断过程：**
```bash
which node                                          # 有路径 ✓
node -v                                             # v26.5.0 ✓
python -c "import shutil; print(shutil.which('node'))"  # 有路径 ✓
echo "process.stdout.write('ok')" | node --permission - # ok ✓
# 但 yt-dlp 仍然 JS runtimes: none — 自动检测在 _NodeJsRuntime 内部失败
```

**解决方法：显式传入 node 路径**

```bash
# 临时（单次命令）
yt-dlp --js-runtimes "node:/home/li.qianyi/envs/hallo/bin/node" ...

# 永久（写入 yt-dlp 配置文件，后续所有命令自动生效）
mkdir -p ~/.config/yt-dlp
echo '--js-runtimes node:/home/li.qianyi/envs/hallo/bin/node' >> ~/.config/yt-dlp/config
```

验证修复：
```bash
yt-dlp --verbose --skip-download --cookies ... URL 2>&1 | grep "JS runtimes"
# 期望输出：[debug] JS runtimes: node-26.5.0
```

**注意：** sbatch 脚本里也要加上 `--js-runtimes` 参数，或确保 `~/.config/yt-dlp/config` 在计算节点上可读。

---

### 17. PyTorch cu130 与集群 CUDA 12.8 驱动不兼容

**报错：**
```
RuntimeError: The NVIDIA driver on your system is too old (found version 12080).
Please update your GPU driver by downloading and installing a new version...
```

**原因：** 集群所有节点（包括 H200）的 CUDA 驱动为 12.8（`12080`），PyTorch `cu130` 编译版本要求驱动 ≥13.0。CUDA 不向前兼容，高版本编译的包无法跑在低版本驱动上。

**解决：** 重装为 cu124（CUDA 12.4 编译，可在 ≥12.4 驱动上运行）：
```bash
pip uninstall torch torchvision xformers -y
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
pip install xformers==0.0.28.post1 --index-url https://download.pytorch.org/whl/cu124
```

验证（在登录节点上 `CUDA available: False` 属于正常，无 GPU）：
```bash
python -c "import torch, torchvision, xformers; print(torch.__version__, xformers.__version__)"
# 期望：2.4.1+cu124  0.0.28.post1
```

---

### 18. xformers 指定版本在 cu124 索引中不存在

**报错：**
```
ERROR: Could not find a version that satisfies the requirement xformers==0.0.27.post2
(from versions: 0.0.28.post1, 0.0.28.post2, 0.0.28.post3, 0.0.29, ...)
```

**原因：** cu124 索引中 xformers 最低版本为 `0.0.28.post1`，`0.0.27.post2` 不存在。

**解决：** 改用最接近的版本：
```bash
pip install xformers==0.0.28.post1 --index-url https://download.pytorch.org/whl/cu124
```

---

### 19. NCCL CUDA 调用失败（单卡训练）

**报错：**
```
torch.distributed.DistBackendError: NCCL error in: ../torch/csrc/distributed/c10d/NCCLUtils.hpp:275,
unhandled cuda error (run with NCCL_DEBUG=INFO for details), NCCL version 2.29.7
ncclUnhandledCudaError: Call to CUDA function failed.
```

**根本原因：** NCCL 2.29.7（随 torch 2.4.1+cu124 捆绑）在集群 CUDA 12.8 驱动上调用某 CUDA API 失败。单卡训练（world_size=1）完全不需要 NCCL，但 SAT 框架在 `hallo3/arguments.py:286` 硬调用 `init_process_group(backend=args.distributed_backend)`，默认值为 `"nccl"`。

**已尝试但无效的方法：**
- `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1` — NCCL 仍初始化，P2P/IB 不是根本原因
- 换 `deepspeed --num_gpus 1` 启动器 — 底层仍调用同一段 Python 初始化代码，报错一样

**解决：** 传 `--distributed_backend gloo` 参数，直接绕过 NCCL：
```bash
torchrun --standalone --nproc_per_node=1 \
    hallo3/train_video.py \
    --base configs/cogvideox_5b_i2v_s1.yaml configs/sft_talkvid.yaml \
    --distributed_backend gloo \
    --seed $RANDOM
```

单卡 world_size=1 无卡间通信，gloo（CPU 通信后端）与 nccl 效果完全等价，且不依赖任何 NCCL CUDA 调用。已更新到 `slurm/run_finetune_s1.sh`。

---

### 20. cuDNN 9.2.0 与驱动 570.86.15 不兼容（CUDNN_STATUS_NOT_INITIALIZED）

**报错：**
```
RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED
```

**现象：** 训练在 iteration 0 的 VAE encoder 第一个 Conv3d 就崩溃，所有 dtype（float32/float16/bfloat16）都失败。

**根本原因：** PyTorch 2.4.1+cu124 自带的 cuDNN 9.2.0 与集群 H200 节点的驱动 570.86.15 不兼容。之前 SLURM 脚本中有 `module load cuda/12.8.0` 和 `module load cuDNN/9.10.2`，加载系统的 cuDNN 9.10.2 覆盖了 PyTorch 自带的 9.2.0，所以能正常运行。去掉这两行后回退到自带的旧版 cuDNN，导致所有 Conv3d 操作失败。

**诊断：**
```bash
# 在 H200 节点上测试（不加载系统 CUDA 模块）
python -c "
import torch
print('cuDNN version:', torch.backends.cudnn.version())  # 92000 (9.2.0)
x = torch.randn(1,3,4,64,64).cuda().bfloat16()
w = torch.randn(128,3,3,3,3).cuda().bfloat16()
y = torch.nn.functional.conv3d(x, w, padding=1)  # FAIL
"
# 禁用 cuDNN 后测试
torch.backends.cudnn.enabled = False  # → OK，确认是 cuDNN 的问题
```

**已尝试但无效的方法：**
- `torch.backends.cudnn.enabled = False` — Python 层面设置但被 SAT/DeepSpeed 初始化重新打开
- `TORCH_CUDNN_V8_API_DISABLED=1` — 导致 `CUDA error: illegal memory access`（SIGABRT）

**解决：** 在 SLURM 脚本中恢复系统 CUDA/cuDNN 模块加载：
```bash
module load cuda/12.8.0
module load cuDNN/9.10.2
```

**教训：** 已验证能跑通的系统模块配置（`module load`）不要随意删除。

---

### 21. FusedEmaAdam CUDA 非法内存访问（optimizer step 崩溃）

**报错：**
```
RuntimeError: CUDA error: an illegal memory access was encountered
```

**调用栈：**
```
deepspeed/runtime/zero/stage_1_and_2.py → optimizer.step()
sat/ops/fused_ema_adam.py:233 → multi_tensor_applier(self.multi_tensor_ema_adam, ...)
deepspeed/ops/adam/multi_tensor_apply.py:17 → return op(...)
RuntimeError: CUDA error: an illegal memory access was encountered
```

**现象：** 训练 forward + backward 正常完成，但在第一次 `optimizer.step()` 时崩溃（SIGABRT）。

**根本原因：** `sat.ops.FusedEmaAdam` 是 SAT 框架自带的自定义 CUDA 算子，未针对 H200（Hopper 架构，compute capability 9.0）编译，运行时产生非法内存访问。与问题 #6 本质相同，但这次出现在 `sft_talkvid.yaml` 配置里（之前只记录了 `sft_s1.yaml`）。

**解决：** 在 `configs/sft_talkvid.yaml` 中将 optimizer 改为标准 PyTorch 的 AdamW：
```yaml
optimizer:
  type: AdamW   # 原来是 sat.ops.FusedEmaAdam
  params:
    lr: 1e-5
    betas: [ 0.9, 0.95 ]
    eps: 1e-8
    weight_decay: 1e-4
```

**注意：** 所有使用 `sat.ops.FusedEmaAdam` 的配置文件（`sft_s1.yaml`、`sft_s2.yaml`、`sft_talkvid.yaml` 等）在 H200 上都需要改成 `AdamW`。

---

### 22. 降分辨率后 einops shape mismatch

**报错：**
```
einops.EinopsError: Shape mismatch, can't divide axis of length 640 in chunks of 1350
Error while processing rearrange-reduction pattern "b (t h w) (c p q) -> b t c (h p) (w q)"
```

**现象：** 将 `sft_talkvid.yaml` 的 `video_size` 从 [480, 720] 降到 [320, 512] 后，训练在 iteration 0 崩溃。

**根本原因：** `configs/cogvideox_5b_i2v_s1.yaml` 中硬编码了 latent 空间的分辨率参数（`latent_width`、`latent_height`、`num_frames`），用于模型 FinalLayer 的 `unpatchify` 操作（`dit_video_concat.py:382`）。改了数据分辨率但没改这些参数，导致 rearrange 时实际 tensor 维度与期望不匹配。

**维度对照：**

| 参数 | 480×720, 49帧 | 320×512, 25帧 |
|------|--------------|--------------|
| `latent_width` | 90 (720/8) | 64 (512/8) |
| `latent_height` | 60 (480/8) | 40 (320/8) |
| `num_frames` | 49 | 25 |
| unpatchify h×w | 30×45 = 1350 | 20×32 = 640 |

**解决：** 在 `cogvideox_5b_i2v_s1.yaml` 中同步修改两处（`network_config` 和 `ref_network_config`）：
```yaml
# 降分辨率时改为：
num_frames: 25
latent_width: 64    # = 新宽度 / 8
latent_height: 40   # = 新高度 / 8
```

**公式：** `latent_width = video_width / 8`，`latent_height = video_height / 8`，`num_frames = max_num_frames`

**注意：** 改回全分辨率训练时，这些参数也要改回 `latent_width: 90, latent_height: 60, num_frames: 49`。

---

## 五、快速参考

### 当前已知可用的训练命令（单卡 H200，2026-08-04 验证）

**前提条件：**
- `configs/sft_talkvid.yaml`：`optimizer.type: AdamW`，`video_size: [480, 720]`，`max_num_frames: 49`
- `configs/cogvideox_5b_i2v_s1.yaml`：`latent_width: 90`，`latent_height: 60`，`num_frames: 49`
- SLURM 脚本中已加载 `module load cuda/12.8.0` + `module load cuDNN/9.10.2`

**Interactive 方式：**
```bash
module load cuda/12.8.0
module load cuDNN/9.10.2
conda activate /home/li.qianyi/envs/hallo
cd /scratch/li.qianyi/hallo3

CUDA_VISIBLE_DEVICES="0" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=1 \
    hallo3/train_video.py \
    --base configs/cogvideox_5b_i2v_s1.yaml configs/sft_talkvid.yaml \
    --seed $RANDOM
```

**Batch 方式：**
```bash
sbatch slurm/run_finetune_s1.sh
```

环境依赖（PyTorch 2.4.1+cu124，xformers 0.0.28.post1）：
```bash
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
pip install xformers==0.0.28.post1 --index-url https://download.pytorch.org/whl/cu124
```

### 当前已知可用的推理命令

```bash
# 确保 inference.yaml 中:
# load: ./stage-1/train-stage-1-07-20-02-38   （父目录，不带迭代号）
# 且父目录下有 latest 文件内容为 "100"

CUDA_VISIBLE_DEVICES="0" bash scripts/inference_long_batch.sh \
    my_inference/input.txt \
    my_inference/outputs/finetuned_100iter/
```
