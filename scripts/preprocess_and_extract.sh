#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --partition=rtx-devel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=p2026_0014_neu
#SBATCH --output=preprocess-%j.out

module load miniforge3
module load cuda
conda activate hallo3

cd ~/hallo3/hallo3

# 建软链接（如果不存在）
ln -sf /scratch/li_qiany_neu/pretrained_models pretrained_models

DATA_ROOT=/scratch/li_qiany_neu/TalkVidTestData
VIDEO_DIR=${DATA_ROOT}/clips_flat
CHECKPOINT=/scratch/li_qiany_neu/pretrained_models/ltx2.5/vae/ltx-2.5-audio-vae-bf16.safetensors

echo "=========================================="
echo "Step 1: data_preprocess.py (face_emb, bbox, wav2vec audio_emb)"
echo "=========================================="
python data_preprocess.py -i ${VIDEO_DIR}

echo ""
echo "=========================================="
echo "Step 2: extract LTX-2 VAE audio embeddings (overwrite wav2vec)"
echo "=========================================="
# data_preprocess.py 会从 mp4 提取 .wav 到 ${DATA_ROOT}/audios/
# 用这些 .wav 文件提取 LTX-2 VAE embeddings，覆盖 audio_emb/
python ../scripts/extract_audio_emb_ltx_vae.py \
    --audio_dir ${DATA_ROOT}/audios \
    --output_dir ${DATA_ROOT}/audio_emb \
    --checkpoint ${CHECKPOINT}

echo ""
echo "=========================================="
echo "Done! Checking results:"
echo "=========================================="
echo "Videos:"
ls ${VIDEO_DIR}/*.mp4 2>/dev/null | wc -l
echo "Face embeddings:"
ls ${DATA_ROOT}/face_emb/*.pt 2>/dev/null | wc -l
echo "Audio embeddings (LTX-2 VAE):"
ls ${DATA_ROOT}/audio_emb/*.pt 2>/dev/null | wc -l
echo ""
echo "Sample audio_emb shape:"
python -c "import torch; e=torch.load('${DATA_ROOT}/audio_emb/$(ls ${DATA_ROOT}/audio_emb/ | head -1)', weights_only=True); print(e.shape)"
