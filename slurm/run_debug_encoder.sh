#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH -o /scratch/li.qianyi/hallo3/logs/debug_encoder_%j.log

# =============================================
#  Encoder Debug: 跑 1 步训练，打印全链路 shape
#  提交: sbatch slurm/run_debug_encoder.sh
#  查看: grep "[AUDIO_DEBUG]" logs/debug_encoder_*.log
# =============================================

HALLO3_DIR="/scratch/li.qianyi/hallo3"
mkdir -p "$HALLO3_DIR/logs"

module load cuda/12.8.0
module load cuDNN/9.10.2
source /home/li.qianyi/.bashrc
conda activate /home/li.qianyi/envs/hallo

cd "$HALLO3_DIR"

echo "=========================================="
echo " Encoder Debug 验证开始: $(date)"
echo " 只跑 1 iteration, 打印所有 [AUDIO_DEBUG]"
echo "=========================================="

# --train-iters 在 sft_talkvid_s2.yaml 里已经是 100，
# 这里用 sed 临时改成 1 步（不改文件，用临时 yaml）
TMP_YAML=$(mktemp /tmp/debug_s2_XXXXXX.yaml)
cat configs/sft_talkvid_s2.yaml | sed 's/train_iters:.*/train_iters: 1/' > "$TMP_YAML"

# 同时把 save_interval 改大，避免保存 checkpoint 浪费时间
sed -i 's/save_interval:.*/save_interval: 99999/' "$TMP_YAML"

echo "临时配置: $TMP_YAML"
echo "--- 配置内容 ---"
cat "$TMP_YAML"
echo "--- 配置结束 ---"

CUDA_VISIBLE_DEVICES="0" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=1 \
    hallo3/train_video.py \
    --base configs/cogvideox_5b_i2v_s2.yaml "$TMP_YAML" \
    --seed 42 \
    2>&1
EXIT_CODE=$?

rm -f "$TMP_YAML"

echo ""
echo "=========================================="
echo " 训练退出码: $EXIT_CODE"
echo "=========================================="

if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ 验证失败! 完整日志:"
    echo "   cat logs/debug_encoder_${SLURM_JOB_ID}.log"
    echo ""
    echo "只看 AUDIO_DEBUG:"
    echo "   grep '[AUDIO_DEBUG]' logs/debug_encoder_${SLURM_JOB_ID}.log"
    exit $EXIT_CODE
fi

echo ""
echo "========== [AUDIO_DEBUG] 汇总 =========="
grep "\[AUDIO_DEBUG\]" /scratch/li.qianyi/hallo3/logs/debug_encoder_${SLURM_JOB_ID}.log || echo "(无 AUDIO_DEBUG 输出)"
echo "========== 汇总结束 =========="

echo ""
echo "✅ 验证通过: $(date)"
