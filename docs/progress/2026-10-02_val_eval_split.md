# Val / Eval Split 准备与训后评估计划

更新日期：2026-10-02

接续 `2026-10-01_4k_train_decord_hang.md`。训练任务已在运行中，本节记录今天完成的 val/eval 数据集准备，以及训完后的评估流程。

---

## 一、已完成：Val / Eval Split 处理

用同一套数据处理流程（`scripts/process_4k_data.py`）处理了 641 条独立视频（未与训练集 2719 条重叠），包含：

- 提取视频帧 (images/)
- 人脸 embedding + mask (face_emb/, face_mask/)
- LTX-2 VAE audio embedding (audio_emb/)，641 条全部成功（ok=641, skip=0, fail=0）
- 生成 caption

最终生成：

| 文件 | 条数 | 路径 |
|------|------|------|
| `talkvid_val.json` | 291 条 | `data/talkvid_val.json` |
| `talkvid_eval.json` | 350 条 | `data/talkvid_eval.json` |

原始 split 元数据：`splits/talkvid_val.json`、`splits/talkvid_eval.json`（含 video id、分辨率等信息）。

---

## 二、Val vs Eval 用途区分

| | Validation | Evaluation |
|--|------------|------------|
| 文件 | `data/talkvid_val.json`（291 条） | `data/talkvid_eval.json`（350 条） |
| 用途 | 训练中/后定量检查，看 loss 趋势和是否过拟合 | 最终效果评估，用于论文对比实验 |
| 时机 | checkpoint 选择阶段 | 确定最佳 checkpoint 后 |
| 指标 | SyncNet confidence score、loss | SyncNet、FVD，以及主观评估 |

---

## 三、训完后的评估步骤

### Step 1：检查 loss 曲线，选 checkpoint

```bash
grep "total loss" /scratch/li_qiany_neu/hallo3_logs/train_4k_*.log | tail -30
ls -lht /scratch/li_qiany_neu/train_output_4k/
```

- 看 loss 收敛情况，不一定取最后一个 checkpoint
- 若 loss 在某步趋平或回升，取趋平前的那个

### Step 2：主观快速验证（val set，5-10 条）

随机取几条 val 视频，用训好的 checkpoint 跑 `sample_video.py`，肉眼看：

- lip-sync 精度（嘴型是否对齐音频）
- 视频质量、人脸自然度
- 与原始 Hallo3（pretrained checkpoint）对比

```bash
python hallo3/sample_video.py \
    --vae_checkpoint /scratch/li_qiany_neu/pretrained_models/ltx2.5/vae/ltx-2.5-audio-vae-bf16.safetensors \
    --checkpoint <训好的 checkpoint 路径> \
    ...
```

### Step 3：SyncNet 定量评估（eval set，350 条）

部署 SyncNet 到集群，批量跑 eval set，计算 confidence score。

对比组：

| 实验 | 编码器 | 数据 | checkpoint |
|------|--------|------|------------|
| Baseline | Wav2Vec2 | 原始 Hallo3 pretrained | pretrained hallo3 |
| Ours | LTX-2 VAE | talkvid_4k_clean (2702 条) | 训好的 4k checkpoint |

指标：SyncNet confidence score（越高越好），FVD（越低越好）。

### Step 4：如结果好，扩数据训练

若新编码器效果优于 baseline，下一步：

- 用 `talkvid_en_8k_quality.json`（8K 条高质量视频）扩数据
- 或直接跑全量 TalkVid 英语数据（见长期计划）

---

## 四、集群同步

需要把以下文件同步到集群（via git pull）：

- `data/talkvid_val.json`
- `data/talkvid_eval.json`
- `splits/talkvid_val.json`
- `splits/talkvid_eval.json`

集群上对应数据（视频、embeddings 等）已在：`/scratch/li_qiany_neu/talkvid_val/` 和 `talkvid_eval/`（或同一目录下按 id 区分）。

---

## 五、待确认

- [ ] SyncNet 工具是否已部署到集群
- [ ] val/eval 视频在集群上的存放路径（确认与 json 里的路径对应）
- [ ] 最终选用哪个 checkpoint（待训练完成后决定）
