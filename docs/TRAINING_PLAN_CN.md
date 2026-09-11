# Hallo3 × TalkVid 训练计划

**研究目标：** 验证用 TalkVid 高质量数据集微调 Hallo3 后，talking head 生成效果是否有提升。

---

## 实验阶段概览

| 阶段 | 数据量 | 迭代数 | 目的 |
|------|--------|--------|------|
| 阶段一 | 42 个视频 | 100 iter | 验证训练流程跑通，无报错 |
| 阶段二 | 42 个视频 | 30000 iter | 验证小数据集训练效果 |
| 阶段三 | 全量数据 | 30000 iter | 验证完整 TalkVid 数据集效果 |

每个阶段完成后做推理，对比生成视频质量，再决定是否进入下一阶段。

---

## 阶段一：100 步流程验证（当前）

### 目的
确认整个训练流程没有问题，能正常保存 checkpoint。

### 配置
```bash
# sft_s1.yaml 当前配置
train_iters: 100
train_data: ["./data/talkvid.json"]   # 42 个视频
```

### 运行
```bash
conda activate /home/li.qianyi/envs/hallo
cd /scratch/li.qianyi/hallo3

CUDA_VISIBLE_DEVICES="0" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=1 \
hallo3/train_video.py \
--base configs/cogvideox_5b_i2v_s1.yaml configs/sft_s1.yaml \
--seed $RANDOM
```

### 验收标准
- [ ] 100 iter 无报错跑完
- [ ] `stage-1/` 目录下有 checkpoint 文件
- [ ] loss 有下降趋势

### checkpoint 确认
```bash
ls stage-1/
```

---

## 阶段二：42 个视频 × 30000 iter

### 目的
用小数据集跑完整训练，快速验证 TalkVid 数据对 Hallo3 的效果。

### 准备
```bash
# 改回 30000 iter
sed -i 's/train_iters: 100/train_iters: 30000/' configs/sft_s1.yaml
```

### 建议用 4 卡提交 batch job（约 15 小时）
```bash
# 申请 4 卡
srun -p sharing --gres=gpu:h200:4 --mem=200G --pty bash

# 或提交 sbatch job（不占用终端）
cat > run_stage1.sh << 'EOF'
#!/bin/bash
#SBATCH -p sharing
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=200G
#SBATCH -o logs/stage1_%j.log

conda activate /home/li.qianyi/envs/hallo
cd /scratch/li.qianyi/hallo3

CUDA_VISIBLE_DEVICES="0,1,2,3" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=4 \
hallo3/train_video.py \
--base configs/cogvideox_5b_i2v_s1.yaml configs/sft_s1.yaml \
--seed 42
EOF

mkdir -p logs
sbatch run_stage1.sh
```

### Stage 2
Stage 1 完成后：
```bash
# 修改 sft_s2.yaml 填入数据路径（同 sft_s1.yaml）
sed -i 's|"",|"./data/talkvid.json",|g' configs/sft_s2.yaml

sbatch run_stage2.sh  # 参考 run_stage1.sh 写法，改脚本为 s2
```

### 推理验证（Stage 2 完成后）
```bash
# 准备推理输入文件
# 格式：文字描述@@参考图片@@音频文件

bash scripts/inference_long_batch.sh \
    my_inference/input_test.txt \
    my_inference/outputs/talkvid_42/
```

### 与原始 Hallo3 对比
用**相同的参考图 + 音频**，分别用：
- 原始 Hallo3 权重推理
- 微调后权重推理

对比生成视频质量。

---

## 阶段三：全量 TalkVid 数据

### 目的
验证完整 TalkVid 数据集（181,752 个视频片段）的训练效果。

### TalkVid 完整数据集说明

本地 JSON 文件位于：
```
/Users/lynnli/Desktop/ResearchNEU/talkvid_datafiles/filtered_video_clips_with_captions.json
```

共 **181,752** 条记录，每条包含：

| 字段 | 含义 |
|------|------|
| `id` | 视频 ID |
| `start-time` / `end-time` | 片段时间段 |
| `description` | 视频文字描述（caption）★ 全量训练时使用 |
| `dover_scores` | 视频质量分数（可用于筛选高质量数据） |
| `cotracker_ratio` | 头部运动幅度 |
| `head_detail` | 头部细节信息 |

**重要：全量训练时使用真实 caption（`description` 字段），替换现在的占位符 "A person talking."**

### 更新转换脚本使用真实 caption

在 `scripts/convert_talkvid_to_hallo3.py` 中，加载 JSON 并替换 caption 写入逻辑：

```python
# 加载 TalkVid caption 映射
import json
caption_map = {}
with open("/path/to/filtered_video_clips_with_captions.json") as f:
    for item in json.load(f):
        caption_map[item["id"]] = item["description"]

# 写 caption 时从 caption_map 查找，找不到才用默认值
caption = caption_map.get(video_id, "A person talking.")
```

转换脚本需要新增 `--caption_json` 参数，全量转换时传入：
```bash
python scripts/convert_talkvid_to_hallo3.py \
    --clips_flat /scratch/li.qianyi/TalkVid/clips_flat_full \
    --output /scratch/li.qianyi/hallo3_data_full \
    --dataset_name talkvid_full \
    --caption_json /path/to/filtered_video_clips_with_captions.json
```

### 数据质量筛选（可选）

可用 `dover_scores` 只保留高质量视频，减少噪声：
```python
# 只保留质量分数 > 阈值的视频
high_quality = [item for item in data if item["dover_scores"] > 0.5]
print(f"高质量视频数量: {len(high_quality)}")  # 待确认实际阈值
```

### 数据转换（全量）
```bash
cd /scratch/li.qianyi/hallo3

python scripts/convert_talkvid_to_hallo3.py \
    --clips_flat /scratch/li.qianyi/TalkVid/clips_flat_full \
    --output /scratch/li.qianyi/hallo3_data_full \
    --dataset_name talkvid_full \
    --caption_json /scratch/li.qianyi/TalkVid/filtered_video_clips_with_captions.json

python hallo3/extract_meta_info.py \
    -r /scratch/li.qianyi/hallo3_data_full \
    -n talkvid_full
```

### 更新训练配置
```bash
sed -i 's|"./data/talkvid.json"|"./data/talkvid_full.json"|g' configs/sft_s1.yaml
sed -i 's|"./data/talkvid.json"|"./data/talkvid_full.json"|g' configs/sft_s2.yaml
```

### 提交训练
```bash
sbatch run_stage1_full.sh  # 同阶段二，改数据路径
```

---

## 效果评估方法

### 定性评估（主观）
用相同输入（图片 + 音频）对比三组结果：
1. **原始 Hallo3**（未微调）
2. **42 视频微调**
3. **全量数据微调**

观察：
- 嘴型同步准确度
- 人脸自然度
- 视频稳定性

### 定量评估（可选）
| 指标 | 含义 |
|------|------|
| Sync-C | 音唇同步置信度（越高越好） |
| Sync-D | 音唇同步距离（越低越好） |
| FID | 生成图像质量 |
| FVD | 生成视频质量 |

---

## 当前已解决的问题记录

| 问题 | 原因 | 解决方法 |
|------|------|---------|
| `pyav==14.0.1` 安装失败 | 包名有误 | 用 `requirements_fixed.txt` |
| `FusedEmaAdam` CUDA 崩溃 | 未为 H200 编译 | 改用 `AdamW` |
| `AdamW is not supported` | DeepSpeed 不认全路径 | 用 `AdamW`（不加前缀）|
| 8 GPU 配置单卡报错 | `CUDA_VISIBLE_DEVICES` 问题 | 改 `--nproc_per_node=1` |
| `pkg_resources` 缺失 | setuptools 版本问题 | `pip install "setuptools<82"` |
