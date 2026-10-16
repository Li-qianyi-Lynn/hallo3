# 2026-10-16：T5 权重缺失与后续计划

承接 [2026-10-09_syncnet_val5_gt_vs_gen.md](2026-10-09_syncnet_val5_gt_vs_gen.md) 和 [../debug/2026-10-15_encoder_replacement_analysis.md](../debug/2026-10-15_encoder_replacement_analysis.md)。

## 一、今日结论

| 项 | 状态 |
|----|------|
| step9800 / step10400 推理 | **阻塞**：T5-v1.1-xxl 模型权重缺失 |
| 根因分析文档 | 已完成（见 debug/2026-10-15_encoder_replacement_analysis.md） |
| T5 tokenizer | 存在（tokenizer_config.json、spiece.model 等） |
| T5 model weights | **缺失**（目录只有 tokenizer，无 pytorch_model.bin / model.safetensors） |
| 解决方案 | 在登录节点用 tmux 下载 ~42GB 权重 |

---

## 二、问题详情

### 2.1 错误

推理 step9800 / step10400 时报错：

```
OSError: Error no file named pytorch_model.bin, tf_model.h5, model.ckpt.index or flax_model.msgpack found in directory ./pretrained_models/t5-v1_1-xxl
```

### 2.2 排查结论

| 检查项 | 结果 |
|--------|------|
| `~/hallo3/hallo3/pretrained_models` 软链 | 正常 → `/scratch/li_qiany_neu/pretrained_models` |
| `/scratch/li_qiany_neu/pretrained_models/t5-v1_1-xxl/` | 只有 4 个 tokenizer 文件 |
| HF cache `~/.cache/huggingface/` | 781KB，无 T5 |
| `TRANSFORMERS_CACHE` / `HF_HOME` | 未设置 |
| 嵌套软链问题 | 已修复（`pretrained_models/pretrained_models` 自引用软链已删） |

T5 tokenizer 是上次手动下载的；模型权重（~42GB）不知何时丢失，推测是 scratch 空间清理或上次推理时未保存到正确位置。

---

## 三、解决方案

在**登录节点**（有网络，计算节点无网络）用 tmux 挂后台下载：

```bash
# 退出计算节点
exit

# 登录节点
tmux new -s t5_download
conda activate hallo3

python - << 'EOF'
from transformers import T5EncoderModel
save_dir = "/scratch/li_qiany_neu/pretrained_models/t5-v1_1-xxl"
print("Downloading T5EncoderModel weights (~42GB)...")
model = T5EncoderModel.from_pretrained("google/t5-v1_1-xxl")
model.save_pretrained(save_dir)
print("Done!")
EOF
```

用 `Ctrl+B D` detach，用 `tmux attach -t t5_download` 查看进度。

下载完验证：

```bash
ls -lh /scratch/li_qiany_neu/pretrained_models/t5-v1_1-xxl/
# 应有 pytorch_model-000xx-of-000xx.bin 分片文件，总计约 42GB
```

---

## 四、下载完后的推理命令

重新申请计算节点后，从 `~/hallo3/hallo3` 运行：

```bash
# step9800
cat > /tmp/infer_step9800.yaml << 'EOF'
args:
  load: /scratch/li_qiany_neu/train_output_4k/train-4k-ltx-vae-10-08-18-40
EOF

TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 \
python sample_video.py \
    --base ../configs/cogvideox_5b_i2v_s2.yaml ../configs/inference.yaml /tmp/infer_step9800.yaml \
    --seed 42 \
    --vae_checkpoint /scratch/li_qiany_neu/pretrained_models/ltx2.5/vae/ltx-2.5-audio-vae-bf16.safetensors \
    --input-file ../my_inference/input_val5.txt \
    --output-dir /scratch/li_qiany_neu/infer_out/step9800

# step10400
cat > /tmp/infer_step10400.yaml << 'EOF'
args:
  load: /scratch/li_qiany_neu/train_output_4k/train-4k-ltx-vae-10-08-22-41
EOF

TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 \
python sample_video.py \
    --base ../configs/cogvideox_5b_i2v_s2.yaml ../configs/inference.yaml /tmp/infer_step10400.yaml \
    --seed 42 \
    --vae_checkpoint /scratch/li_qiany_neu/pretrained_models/ltx2.5/vae/ltx-2.5-audio-vae-bf16.safetensors \
    --input-file ../my_inference/input_val5.txt \
    --output-dir /scratch/li_qiany_neu/infer_out/step10400
```

---

## 五、今日修复汇总

今天推理过程中踩到的坑和修复：

| 错误 | 原因 | 修复 |
|------|------|------|
| T5 权重缺失 | `save_pretrained()` OOM killed | `huggingface-cli download` 直接复制 cache，不加载进内存 |
| ltx2.5 VAE 缺失 → fallback wav2vec | scratch 文件丢失 | `huggingface-cli download Lightricks/LTX-2.5 vae/...` |
| CUDA 未初始化 → triton RuntimeError | 新 shell 未加载 cuda 模块，集群升级 12.8→13.1.1 | `module load cuda/13.1.1` |
| insightface `assert 'detection' in self.models` | insightface 版本不再自动扫描子目录，`name=""` 找不到 `buffalo_l/` | `sed -i 's/name="",/name="buffalo_l",/'` image_processor.py（两处） |

### insightface 修复详情

`hallo3/sgm/utils/image_processor.py` line 56 和 line 162：

```python
# 改前
name="",
# 改后
name="buffalo_l",
```

**注意**：只在集群 `train7/new` 分支上用 `sed -i` 直接改，**本地代码库未同步**。需要在本地也应用同一改动后 commit，或在集群上 commit 后 pull。

### 推理前必须执行的环境设置（新 shell 每次都要）

```bash
module load cuda/13.1.1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
conda activate hallo3
cd ~/hallo3/hallo3
```

---

## 六、后续任务（按优先级）

1. **T5 下载完后**：跑 step9800 + step10400 推理，SyncNet 打分，汇总 5 条 val 跨 checkpoint 趋势
2. **扩至 200 条 val**（导师要求）：用 `scripts/make_val_infer_input.py --n 200 --seed 0` 生成 `my_inference/input_val200.txt`，对选定 checkpoint 批量推理打分 (这点可以暂时不用)

---

