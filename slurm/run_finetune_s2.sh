#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH -o /scratch/li.qianyi/hallo3/logs/finetune_s2_%j.log

HALLO3_DIR="/scratch/li.qianyi/hallo3"
CKPT_DIR="$HALLO3_DIR/stage-2"

mkdir -p "$HALLO3_DIR/logs"

# ==== 激活环境 ====
module load cuda/12.8.0
module load cuDNN/9.10.2
source /home/li.qianyi/.bashrc
conda activate /home/li.qianyi/envs/hallo

cd "$HALLO3_DIR"

echo "开始 Stage 2 训练: $(date)"

CUDA_VISIBLE_DEVICES="0" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=1 \
    hallo3/train_video.py \
    --base configs/cogvideox_5b_i2v_s2.yaml configs/sft_talkvid_s2.yaml \
    --seed $RANDOM
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "❌ Stage 2 训练异常退出 (exitcode: $TRAIN_EXIT)，请检查日志"
    exit $TRAIN_EXIT
fi

echo "✅ Stage 2 训练完成: $(date)"
echo "Checkpoint 路径: $CKPT_DIR"
