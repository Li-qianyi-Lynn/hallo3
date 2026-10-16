# 2026-10-08：停止训练、val 集核对与 checkpoint 推理准备

承接 [2026-10-08_loss_analysis.md](2026-10-08_loss_analysis.md) 和 [2026-10-07_queue_devel_syncnet.md](2026-10-07_queue_devel_syncnet.md)。

## 一、今日结论

| 项 | 结果 |
|----|------|
| 训练 | 决定不跑满 13600 步，在约 11000 步停止（loss 在 2500 步之后基本是平的） |
| val / eval 来源 | 100% 来自 8K 集，与 4K 训练集零重叠，说话人（源视频）也零重叠，用于选 checkpoint 和最终评估没有泄漏 |
| 推理输入 | 已用 `scripts/make_val_infer_input.py` 从 val 里随机挑 5 条（`--seed 0`），生成 `my_inference/input_val5.txt` |
| 下一步 | 在 GPU 节点上用累计 7000 步的 checkpoint 对这 5 条做推理，肉眼看效果 |

---

## 二、训练进度与停止决定

### 累计步数

各目录 `latest` 相加，截至本次检查 **10800**

### 为什么停在这里

- 数据集只有 2702 条 clip（约 3.8 小时），13600 步约 10 个 epoch，小数据集有过拟合风险
- loss 从约 2500 步起稳定在 0.05–0.09，没有持续下降，但 loss 平不能证明继续训练没用，效果要靠 SyncNet / 人工评估判断
- 如果评估显示分数一直在涨，续训路径仍可用，还剩约 2600 步没跑

### 保留作为评估候选的 checkpoint（不要再精简）

| 累计步数 | 目录 |
|----------|------|
| 1300 | `10-01-19-00/600` |
| 3400 | `10-02-00-44/2100` |
| 4900 | `10-02-18-08/1500` |
| 7000 | `10-06-23-37/2100` |
| 9800 | `10-08-18-40/600` |
| 10400 | `10-08-22-41/600` |
| ~11000 | `10-09-01-12/600`（这一轮结束后） |

`10-09-01-12/200` 和 `400` 可以删。8000、8600、9200 之前已经删掉，所以 6500–10000 区间里目前只有 7000 和 9800 两个候选。

### 其他

- 用 wandb 看曲线：`WANDB=1 NO_RESUBMIT=1 bash slurm/run_train_4k.sh ...`；首次报 `Descriptors cannot be created directly`（protobuf 与 tensorboard 不兼容），脚本里开 wandb 时已加 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`。**修复后是否真的跑通未确认**
- 每一轮续训在 wandb 里是单独一条曲线（步数从 0 重新计）
- `10-09-00-44` 是那次 wandb 报错留下的空目录，没有 `latest`，不影响统计

---

## 三、val / eval 集来源核对（用 `splits/` 里的 id 比对）

| 对比 | 结果 |
|------|------|
| val（291 条）∩ 8K 集 | 291，100% |
| eval（350 条）∩ 8K 集 | 350，100% |
| val ∩ 4K 训练集 | 0 |
| eval ∩ 4K 训练集 | 0 |
| val 的 48 个源视频 / eval 的 45 个源视频 与训练集的 1718 个源视频 | 0 个共享 |
| val 与 eval 共享源视频 | 0 个 |
| 4K 训练集 ∩ 8K 集 | 197 条（4.9%） |

结论：val 用于选 checkpoint、eval 用于最终评估，没有数据泄漏，分工不变（val 现在用，eval 选定 checkpoint 之后才用）。

### 扩数据时必须注意

**val 和 eval 的 641 条都在 8K 集里。** 之后如果用 `talkvid_en_8k_quality.json` 扩数据训练，必须先剔除这 641 条，否则评估失效。另外 8K 与现有 4K 训练集有 197 条重叠，不算新数据。

---

## 四、推理输入（val 5 条）

生成命令（在仓库根目录 `~/hallo3`，注意不是 `~/hallo3/hallo3`）：

```bash
python scripts/make_val_infer_input.py --json data/talkvid_val.json --n 5 --seed 0 --out my_inference/input_val5.txt
```

结果：写入 5 条，跳过缺文件 0 条，对应关系在 `my_inference/input_val5.txt.map.tsv`（含真值视频路径）。

### 选中的 5 个 clip

| 序号 | clip 名 |
|------|---------|
| 1 | `videovideo4610_qJNcXjKzC0I-scene4-scene1` |
| 2 | `videovideoGXJQQqfB-ew-scene203-scene5` |
| 3 | `videovideo4V21SWg1V9Q-scene5-scene2` |
| 4 | `videovideo_nllSIHloqz8-scene40-scene1` |
| 5 | `videovideo988_Rx-xZfFjVDQ-scene5-scene4` |

### 需要订正和注意的点

- **val 的素材目录是 `/scratch/li_qiany_neu/talkvid_8k/`**（`images/`、`audios/` 都在这里），不是我之前假设的 `talkvid_val/`。路径以 json 里写的为准，已经正确
- 5 条的 caption 都是 `A person talking.`，是脚本里没找到 caption 文件时的默认值。训练集用的 caption 和这个是否一致没有核对
- 图片是每个 clip 的第一帧（`000000.jpg`），音频是 `.m4a`，推理侧能否直接读 m4a **未验证**，不行的话先转 wav
- 固定 `--seed 0`，以后换 checkpoint 一律用同一批 5 条

---

## 五、推理命令（待在 GPU 节点上验证）

申请节点（checkpoint 文件 73GB，内存给 128G）：

```bash
srun --partition=rtx-devel --nodes=1 --gpus=1 --cpus-per-task=8 --mem=128G --time=01:00:00 --account=p2026_0014_neu --pty bash
```

运行（在节点上，工作目录 `~/hallo3/hallo3`，里面有 `pretrained_models` 软链接）：

```bash
conda activate hallo3
cd ~/hallo3/hallo3

cat > /tmp/infer_7000.yaml << 'EOF'
args:
  load: /scratch/li_qiany_neu/train_output_4k/train-4k-ltx-vae-10-06-23-37
EOF

TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python sample_video.py \
    --base ../configs/cogvideox_5b_i2v_s2.yaml ../configs/inference.yaml /tmp/infer_7000.yaml \
    --seed 42 \
    --vae_checkpoint /scratch/li_qiany_neu/pretrained_models/ltx2.5/vae/ltx-2.5-audio-vae-bf16.safetensors \
    --input-file ../my_inference/input_val5.txt \
    --output-dir /scratch/li_qiany_neu/infer_out/step7000
```

要点：
- `load` 用叠加的 yaml 覆盖 `inference.yaml` 里的原版权重路径，SAT 读取该目录下 `latest` 指向的 checkpoint
- 必须加 `--vae_checkpoint`，否则推理会退回 wav2vec，与训练用的 LTX-2 VAE 不一致
- 必须加 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`，与训练一致
- 换 checkpoint：改 yaml 里的 `load` 和 `--output-dir`，输入文件不变

### 未验证

- `inference.yaml` 里引用的 `pretrained_models/face_analysis`、`wav2vec` 等子目录是否存在
- m4a 能否直接读取
- 单条采样耗时（没有数据）
- 这是推理链路第一次在新集群、新编码器上完整运行，可能有报错

---

## 六、下一步

1. 7000 步 checkpoint 对 5 条 val 推理，肉眼看口型和画质，拉到本地看
2. 同样的 5 条，对 4900、9800、10400、~11000 各跑一遍，看趋势
3. SyncNet 在 val 集（291 条）上对候选 checkpoint 打分，选最高的；先确认 SyncNet 对生成视频的帧率和人脸处理
4. 选定后在 eval 集（350 条）上和原版 Hallo3 对比
5. 补 A 组对照：原 wav2vec 编码器在同一份 2702 条数据上训同样步数
6. 扩数据前，先从 8K 里剔除 val / eval 的 641 条
