#!/bin/bash
#
# 音频 embedding 重提取 — 只需 CPU，不需要 H200
#
# 选项 A（推荐）：CPU 节点，4 并行，约 4~8 小时
#   #SBATCH -p short
#   #SBATCH --cpus-per-task=8
#   （去掉 --gres 行）
#
# 选项 B：任意小 GPU，4 并行，约 1~2 小时
#   #SBATCH -p gpu
#   #SBATCH --gres=gpu:1        # 任意 GPU，不指定型号
#
# 默认使用选项 A（CPU），取消注释选项 B 可切换到 GPU
#
#SBATCH -p gpu
#SBATCH --gres=gpu:1         # 任意 GPU，不指定型号
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-3          # 4 并行 job，各处理约 15,000 个视频
#SBATCH -o /scratch/li.qianyi/hallo3/logs/reextract_audio_%A_%a.log

HALLO3_DIR="/scratch/li.qianyi/hallo3"
DATA_DIR="/scratch/li.qianyi/hallo3_data"

mkdir -p "$HALLO3_DIR/logs"

source /home/li.qianyi/.bashrc
conda activate /home/li.qianyi/envs/hallo

cd "$HALLO3_DIR"

module load cuda/12.8.0
module load cuDNN/9.10.2

echo "开始重提取音频 embedding: rank=${SLURM_ARRAY_TASK_ID}/4, $(date)"

python scripts/reextract_audio_emb.py \
    --data_dir "$DATA_DIR" \
    --wav2vec_model_path pretrained_models/wav2vec/wav2vec2-base-960h \
    --parallelism 4 \
    --rank "$SLURM_ARRAY_TASK_ID" \
    --skip_existing

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ rank=${SLURM_ARRAY_TASK_ID} 异常退出 (exitcode: $EXIT_CODE)"
    exit $EXIT_CODE
fi

echo "✅ rank=${SLURM_ARRAY_TASK_ID} 完成: $(date)"
