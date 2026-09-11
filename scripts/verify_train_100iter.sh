#!/bin/bash
#SBATCH --job-name=verify100
#SBATCH --partition=rtx-devel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=p2026_0014_neu
#SBATCH --output=verify100-%j.out

module load miniforge3
module load cuda
source activate hallo3

cd ~/hallo3/hallo3
ln -sf /scratch/li_qiany_neu/pretrained_models pretrained_models

echo "=========================================="
echo "100 iteration verification with LTX-2 VAE audio encoder"
echo "=========================================="

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=1 \
  train_video.py \
  --base configs/cogvideox_5b_i2v_s2.yaml configs/sft_s2_verify.yaml \
  --seed 42

echo "=========================================="
echo "Done!"
echo "=========================================="
