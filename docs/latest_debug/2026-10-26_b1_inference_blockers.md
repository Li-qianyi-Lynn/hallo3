# 2026-10-26：B1 Baseline 推理进度 & Blockers

承接 [2026-10-23_rp.md](../research_plan/2026-10-23_rp.md)。

## 目标

跑 B1（原版 Hallo3 pretrained，不微调）在 val200 上推理，获得 SyncNet baseline 分数。

## 环境

- 集群节点: `a0018`
- 代码: `~/hallo3_orig`（main 分支 worktree）
- Conda 环境: `hallo3`

## 已解决的问题

| # | 问题 | 原因 | 解决方式 |
|---|------|------|----------|
| 1 | `FileNotFoundError: .../hallo3/1/mp_rank_00_model_states.pt` | checkpoint 未下载 | `huggingface-cli download fudan-generative-ai/hallo3 --include "hallo3/*"` 下载到 scratch，然后 symlink |
| 2 | `OSError: no file named pytorch_model.bin ... in wav2vec2-base-960h` | wav2vec 模型缺失 | `huggingface-cli download facebook/wav2vec2-base-960h` |
| 3 | `AssertionError: 'detection' in self.models` (InsightFace) | `buffalo_l/` 目录为空 | 从 insightface releases 下载 `buffalo_l.zip` |
| 4 | `AssertionError: audio_path` 不存在 | `input_val200.txt` 路径拼接错误 (`videovideo` 重复) | `sed -i 's/videovideo/video/g'` 修复（但后来发现音频文件本身也不在） |

## 当前 Blocker

**`/scratch/` 上的原始音频和视频数据被自动清除。**

- `/scratch/li_qiany_neu/talkvid_8k/audios/` — 空
- `/scratch/li_qiany_neu/talkvid_8k/videos/` — 空
- 原因: AICR 集群 `/scratch/` policy 是 **30 天自动清除**不活跃文件
- 预处理文件还在 (`audio_emb/`, `images/`, `face_emb/`, `face_mask/`)，因为训练时被访问过所以时间戳刷新了
- 推理需要原始音频文件（`.m4a` 或 `.wav`），preprocess 后的 `.pt` 不够

## 已恢复数据（2026-10-29）

音频和视频数据已重新导入到 `/scratch/li_qiany_neu/talkvid_8k/`（4000+ 条，从 8k 筛选出来的）。

## B1 推理命令

```bash
cd ~/hallo3_orig
conda activate hallo3

WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 \
python hallo3/sample_video.py \
  --base ./configs/cogvideox_5b_i2v_s2.yaml ./configs/inference.yaml \
  --seed 42 \
  --input-file ~/hallo3/my_inference/input_val200.txt \
  --output-dir /scratch/li_qiany_neu/infer_out/orig_hallo3_val200
```

- 代码: `~/hallo3_orig`（main 分支 worktree，不需要 git checkout）
- 断点续跑: `sample_video.py` 已加 skip 逻辑，检测 `000000_with_audio.mp4` 存在则跳过
- 输出: `/scratch/li_qiany_neu/infer_out/orig_hallo3_val200/<clip_name>/000000_with_audio.mp4`

### 查看进度

```bash
# 已完成多少条
ls /scratch/li_qiany_neu/infer_out/orig_hallo3_val200/*/000000_with_audio.mp4 2>/dev/null | wc -l

# 剩余
echo "剩余: $((200 - $(ls /scratch/li_qiany_neu/infer_out/orig_hallo3_val200/*/000000_with_audio.mp4 2>/dev/null | wc -l))) 条"
```

## 还需要做的

- [ ] B1 推理跑完 200 条
- [ ] 跑 SyncNet 打分
- [ ] 以后重要数据放 `/work/` 或定期 `touch` scratch 文件防清除

## Pretrained Models 状态

| 模型 | 路径 | 状态 |
|------|------|------|
| Hallo3 checkpoint | `pretrained_models/hallo3/1/mp_rank_00_model_states.pt` | ✅ symlink → scratch (29.1GB) |
| wav2vec2-base | `pretrained_models/wav2vec/wav2vec2-base-960h/` | ✅ 已下载 |
| InsightFace buffalo_l | `pretrained_models/face_analysis/models/buffalo_l/` | ✅ 已修复 |
| Audio separator | `pretrained_models/audio_separator/` | ✅ (symlink → scratch) |
| CogVideoX VAE | `pretrained_models/cogvideox-5b-i2v-sat/vae/3d-vae.pt` | ✅ 已加载 |
| T5 text encoder | `pretrained_models/google/t5-v1_1-xxl/` | 未确认 |



