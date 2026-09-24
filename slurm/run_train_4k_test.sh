#!/bin/bash
#SBATCH --job-name=train-4k-test
#SBATCH --partition=b200-batch
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --account=p2026_0014_neu
#SBATCH --output=/scratch/li_qiany_neu/hallo3_logs/train_4k_test_%j.log

mkdir -p /scratch/li_qiany_neu/hallo3_logs

module load miniforge3
module load cuda
source activate hallo3

cd ~/hallo3/hallo3
ln -sf /scratch/li_qiany_neu/pretrained_models pretrained_models

echo "=========================================="
echo " 4K LTX-2 VAE Training TEST (2x B200)"
echo " Start: $(date)"
echo " GPUs: $SLURM_GPUS_ON_NODE"
echo "=========================================="

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=2 \
    train_video.py \
    --base configs/cogvideox_5b_i2v_s2.yaml configs/sft_4k_train.yaml \
    --seed 42

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
