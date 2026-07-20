# 使用 TalkVid 数据集运行 Hallo3

本文档介绍在 Slurm 集群上将 TalkVid 数据转换为 Hallo3 格式，并完成微调和推理的完整流程。

---

## 0. 前置准备

### 创建 conda 环境

```bash
# 如果目录已存在但不是 conda 环境，先删除
rm -rf /home/li.qianyi/envs/hallo

# 创建环境
conda create --prefix /home/li.qianyi/envs/hallo python=3.10 -y
conda activate /home/li.qianyi/envs/hallo

# 安装依赖（requirements.txt 中 pyav==14.0.1 有误，先去掉）
grep -v "pyav==14.0.1" requirements.txt > requirements_fixed.txt
pip install -r requirements_fixed.txt
```

### 申请 GPU 节点

```bash
# 查看当前空闲 GPU
sinfo -o "%P %G %N %t" | grep idle

  srun -p sharing --gres=gpu:h100:1 --mem=60G --pty bash                                                     

# 注意：sharing 分区不支持 --time 参数，去掉即可
```

### 验证环境

```bash
conda activate /home/li.qianyi/envs/hallo
python -c "import torch, cv2; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 1. TalkVid 数据结构说明

TalkVid 预处理后的数据存放在 `clips_flat/`：

```
/scratch/li.qianyi/TalkVid/clips_flat/
├── videos-crop/            # 裁剪后的人脸视频 (.mp4)
├── new_face_info/          # 每帧人脸检测结果 (.pt)
│                           #   list[帧] → list[人脸] → {bbox, 关键点, embedding}
└── short_clip_aud_embeds/  # 音频 embedding (.pt)
                            #   dict{'global_embeds': Tensor[T, 1, 768]}
```

检查各目录文件数量（理论上应一致）：
```bash
cd /scratch/li.qianyi/TalkVid/clips_flat
ls videos-crop | wc -l            # 45
ls new_face_info | wc -l          # 42（缺 3 个，测试阶段可忽略）
ls short_clip_aud_embeds | wc -l  # 45
```

查找缺失的人脸信息文件：
```bash
comm -23 <(ls videos-crop | sed 's/\.[^.]*$//' | sort) \
         <(ls new_face_info | sed 's/\.[^.]*$//' | sort)
```

---

## 2. 数据转换：TalkVid → Hallo3 格式


python scripts/convert_talkvid_to_hallo3.py --clips_flat /scratch/li.qianyi/TalkVid/clips_flat --output /scratch/li.qianyi/hallo3_data --dataset_name talkvid
```

脚本对每个视频做的事情：

| 步骤 | 输入 | 输出 |
|------|------|------|
| 复制视频 | `videos-crop/*.mp4` | `videos/*.mp4` |
| 抽取帧 | `videos-crop/*.mp4` | `images/*/000000.jpg ...` |
| 人脸 embedding | `new_face_info/*.pt` → 对所有帧取平均 | `face_emb/*.pt`（Tensor[512]） |
| 人脸 mask | `new_face_info/*.pt` → bbox → 生成二值图 | `face_mask/*.png` |
| 音频 embedding | `short_clip_aud_embeds/*.pt` → 提取 tensor | `audio_emb/*.pt`（Tensor[T,1,768]） |
| 文字描述 | — | `caption/*.txt`（"A person talking."） |

转换后输出目录结构：
```
/scratch/li.qianyi/hallo3_data/
├── videos/
├── images/
├── face_emb/
├── face_mask/
├── audio_emb/
└── caption/
```

---

## 3. 生成训练索引文件（Meta JSON）

```bash
cd /hallo3

python hallo3/extract_meta_info.py \
    -r /scratch/li.qianyi/hallo3_data \
    -n talkvid
```

生成 `data/talkvid.json`，这是 Hallo3 训练时读取的数据索引。

验证生成结果：
```bash
python -c "import json; d=json.load(open('data/talkvid.json')); print(f'共 {len(d)} 条样本')"
```

---

## 4. 微调（Fine-tuning）

编辑 `configs/sft_s1.yaml`，填入数据路径：

```yaml
train_data: [
  "./data/talkvid.json",
]
valid_data: [
  "./data/talkvid.json",
]
```

运行 Stage 1：
```bash
bash scripts/finetune_multi_gpus_s1.sh
```

Stage 1 完成后运行 Stage 2：
```bash
bash scripts/finetune_multi_gpus_s2.sh
```

checkpoint 每 500 步自动保存到 `./stage-1/` 和 `./stage-2/`。

---

## 5. 推理（Inference）

用我自己的图
### 运行推理

```bash
bash scripts/inference_long_batch.sh \
    my_inference/input_talkvid.txt \
    my_inference/outputs/
```

生成视频保存在 `my_inference/outputs/`。

---

## 常见报错解决

| 报错 | 原因 | 解决方法 |
|------|------|---------|
| `pyav==14.0.1` 安装失败 | PyPI 上没有这个包 | 用 `requirements_fixed.txt` 安装 |
| `Killed`（加载 .pt 文件时） | 登录节点内存限制 | 在计算节点上运行 |
| `prefix already exists` | conda 环境目录已存在 | `rm -rf /home/li.qianyi/envs/hallo` 后重建 |
| `DirectoryNotACondaEnvironmentError` | 目录存在但不是 conda 环境 | 同上，先删除目录 |
| `srun: Requested time limit is invalid` | sharing 分区不支持 --time | 去掉 `--time` 参数 |
| new_face_info 少 3 个文件 | 对应视频人脸检测失败 | 测试阶段可忽略，42/45 正常 |
