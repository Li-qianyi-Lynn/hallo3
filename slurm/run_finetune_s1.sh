#!/bin/bash
#SBATCH -p sharing
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=200G
#SBATCH --cpus-per-task=32
#SBATCH -o /scratch/li.qianyi/hallo3/logs/finetune_s1_%j.log

HALLO3_DIR="/scratch/li.qianyi/hallo3"
THIS_SCRIPT="$HALLO3_DIR/slurm/run_finetune_s1.sh"
CKPT_DIR="$HALLO3_DIR/stage-1"

mkdir -p "$HALLO3_DIR/logs"

# ==== 防重复提交 ====
safe_resubmit() {
    already=$(squeue -u li.qianyi -h -o "%j" 2>/dev/null | grep -c "run_finetune" || true)
    if [ "$already" -ge 1 ]; then
        echo "队列中已有 run_finetune job，跳过重提交"
    else
        sbatch "$THIS_SCRIPT"
    fi
}

# ==== SIGTERM 捕获：时间到前 60s 保存并重提交 ====
resubmit() {
    echo "⚠️  SIGTERM received, resubmitting: $(date)"
    safe_resubmit
    kill 0
    exit 0
}
trap resubmit SIGTERM

# ==== 激活环境 ====
module load cuda/12.8.0
module load cuDNN/9.10.2
source /home/li.qianyi/.bashrc
conda activate /home/li.qianyi/envs/hallo

cd "$HALLO3_DIR"

# ==== 检测最新 checkpoint，有则续训 ====
LATEST_CKPT=$(ls -d "$CKPT_DIR"/[0-9]* 2>/dev/null | sort -V | tail -1)
if [ -n "$LATEST_CKPT" ]; then
    echo "Resuming from checkpoint: $LATEST_CKPT"
    # 用 sed 临时替换 load 路径
    sed -i "s|load:.*|load: $LATEST_CKPT|" configs/sft_talkvid.yaml
else
    echo "No checkpoint found, starting from pretrained model"
fi

echo "开始训练: $(date)"

CUDA_VISIBLE_DEVICES="0,1,2,3" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=4 \
    hallo3/train_video.py \
    --base configs/cogvideox_5b_i2v_s1.yaml configs/sft_talkvid.yaml \
    --seed $RANDOM

echo "✅ 本轮训练完成: $(date)"

# ==== 检查是否训练完成（stage-1/100/ 存在则完成）====
if [ -d "$CKPT_DIR/100" ]; then
    echo "🎉 100 iterations 完成！"
else
    echo "未完成，自动重提交..."
    safe_resubmit
fi
