#!/bin/bash
#SBATCH --job-name=train-4k
#SBATCH --partition=b200-batch
#SBATCH --gpus=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --account=p2026_0014_neu
#SBATCH --output=/scratch/li_qiany_neu/hallo3_logs/train_4k_%j.log

mkdir -p /scratch/li_qiany_neu/hallo3_logs

module load miniforge3
module load cuda
source activate hallo3

cd ~/hallo3/hallo3
ln -sf /scratch/li_qiany_neu/pretrained_models pretrained_models

NGPUS=$(nvidia-smi -L | wc -l)
echo "=========================================="
echo " 4K LTX-2 VAE Training"
echo " Start: $(date)"
echo " GPUs detected: $NGPUS"
echo "=========================================="

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TORCH_DISABLE_ADDR2LINE=1 \
torchrun --standalone --nproc_per_node=$NGPUS \
    train_video.py \
    --base ../configs/cogvideox_5b_i2v_s2.yaml ../configs/sft_4k_train.yaml \
    --seed $RANDOM

TRAIN_EXIT=$?

echo "=========================================="
echo " Exit code: $TRAIN_EXIT"
echo " End: $(date)"
echo "=========================================="

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "Training failed, check log above"
    exit $TRAIN_EXIT
fi

echo "Checkpoints: /scratch/li_qiany_neu/train_output_4k/"
ls -lh /scratch/li_qiany_neu/train_output_4k/
