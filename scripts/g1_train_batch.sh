#!/bin/bash
#SBATCH --job-name=g1_train
#SBATCH --partition=rtx-batch
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --account=p2026_0014_neu
#SBATCH --output=/scratch/li_qiany_neu/hallo3_logs/g1_train_%j.log
#SBATCH --error=/scratch/li_qiany_neu/hallo3_logs/g1_train_%j.log

echo "=== G1 Training (regression-init adapter) ==="
echo "Node: $(hostname), GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

module load cuda/13.1.1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
source activate hallo3

cd ~/hallo3/hallo3

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=2 \
    train_video.py \
    --base ../configs/cogvideox_5b_i2v_s2.yaml ../configs/sft_g1_train.yaml \
    --seed 42

echo "=== Done: $(date) ==="
echo "Latest checkpoint:"
cat /scratch/li_qiany_neu/train_output_g1/train-g1-*/latest 2>/dev/null
