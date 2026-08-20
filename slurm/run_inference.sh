#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH -o /scratch/li.qianyi/hallo3/logs/inference_%j.log

HALLO3_DIR="/scratch/li.qianyi/hallo3"

mkdir -p "$HALLO3_DIR/logs"
mkdir -p "$HALLO3_DIR/outputs"

# ==== 激活环境 ====
module load cuda/12.8.0
module load cuDNN/9.10.2
source /home/li.qianyi/.bashrc
conda activate /home/li.qianyi/envs/hallo

cd "$HALLO3_DIR"

echo "开始推理: $(date)"

CUDA_VISIBLE_DEVICES="0" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash scripts/inference_long_batch.sh \
    my_inference/input_20260819_orig_woman_snow_0001.txt \
    outputs/

INFER_EXIT=$?

if [ $INFER_EXIT -ne 0 ]; then
    echo "❌ 推理异常退出 (exitcode: $INFER_EXIT)，请检查日志"
    exit $INFER_EXIT
fi

echo "✅ 推理完成: $(date)"
echo "输出目录: $HALLO3_DIR/outputs/"
