#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-3          # 4 并行 job，各处理 1/4 的视频
#SBATCH -o /scratch/li.qianyi/hallo3/logs/reextract_audio_%A_%a.log

HALLO3_DIR="/scratch/li.qianyi/hallo3"
DATA_DIR="/scratch/li.qianyi/hallo3_data"

mkdir -p "$HALLO3_DIR/logs"

module load cuda/12.8.0
module load cuDNN/9.10.2
source /home/li.qianyi/.bashrc
conda activate /home/li.qianyi/envs/hallo

cd "$HALLO3_DIR"

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
