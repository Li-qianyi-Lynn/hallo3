#!/bin/bash
#SBATCH --job-name=train-4k-2gpu
#SBATCH --partition=b200-batch
#SBATCH --gres=gpu:b200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --account=p2026_0014_neu
#SBATCH --output=/scratch/li_qiany_neu/hallo3_logs/train_4k_%j.log
#
# 2×B200 × 24h 分段训练 (4K / LTX-2 VAE)
#   2719 clips / 2 GPU ≈ 1360 steps/epoch
#   10 epochs ≈ 13600 steps ≈ 34h @ ~9s/step（B200 可能更快）
#   Session 1: 0 → ~9600  (~24h)
#   Session 2: ~9600 → 13600 (~10h，超时自动续训)
# 超时前 3 分钟自动 sbatch 下一轮；启动时自动找最新 checkpoint 续训。
# 提交: sbatch slurm/run_train_4k.sh

REPO_DIR="${HOME}/hallo3"
THIS_SCRIPT="${REPO_DIR}/slurm/run_train_4k.sh"
SAVE_ROOT="/scratch/li_qiany_neu/train_output_4k"
PRETRAINED="/scratch/li_qiany_neu/pretrained_models/hallo3"
LOG_DIR="/scratch/li_qiany_neu/hallo3_logs"
TRAIN_ITERS=13600
JOB_NAME="train-4k-2gpu"

mkdir -p "$LOG_DIR" "$SAVE_ROOT"

# ---- 超时前自动重提交（USR1，不杀当前训练）----
RESUBMITTED=0
safe_resubmit() {
    if [ "$RESUBMITTED" = "1" ]; then
        return
    fi
    already=$(squeue -u "$USER" -h -n "$JOB_NAME" -o "%i" 2>/dev/null \
        | grep -v "^${SLURM_JOB_ID}$" | wc -l | tr -d ' ')
    if [ "${already:-0}" -ge 1 ]; then
        echo "[resubmit] 队列中已有 ${JOB_NAME}，跳过"
        return
    fi
    echo "[resubmit] sbatch ${THIS_SCRIPT}  $(date)"
    sbatch "$THIS_SCRIPT"
    RESUBMITTED=1
}

on_time_warning() {
    echo "=========================================="
    echo " USR1: 距超时约 3 分钟，排队下一轮  $(date)"
    echo "=========================================="
    safe_resubmit
}
trap on_time_warning USR1

# ---- 找最新 SAT checkpoint（{save}/{experiment_name}-{ts}/latest）----
find_latest_ckpt() {
    local best_dir="" best_iter=0 f dir iter
    while IFS= read -r f; do
        [ -f "$f" ] || continue
        dir=$(dirname "$f")
        iter=$(tr -d '[:space:]' < "$f")
        if [[ "$iter" =~ ^[0-9]+$ ]] && [ "$iter" -ge "$best_iter" ]; then
            best_iter=$iter
            best_dir=$dir
        fi
    done < <(find "$SAVE_ROOT" -name latest -type f 2>/dev/null)
    echo "${best_dir}|${best_iter}"
}

CKPT_INFO=$(find_latest_ckpt)
LATEST_EXP="${CKPT_INFO%%|*}"
LATEST_ITER="${CKPT_INFO##*|}"
LATEST_ITER="${LATEST_ITER:-0}"

if [ -n "$LATEST_EXP" ] && [ "$LATEST_ITER" -ge "$TRAIN_ITERS" ]; then
    echo "训练已完成: ${LATEST_EXP}  iter=${LATEST_ITER}/${TRAIN_ITERS}"
    exit 0
fi

module load miniforge3
module load cuda
source activate hallo3

cd "${REPO_DIR}/hallo3"
ln -sf /scratch/li_qiany_neu/pretrained_models pretrained_models

OVERLAY="${LOG_DIR}/sft_4k_overlay_${SLURM_JOB_ID}.yaml"
BASE_YAMLS="../configs/cogvideox_5b_i2v_s2.yaml ../configs/sft_4k_train.yaml"
if [ -n "$LATEST_EXP" ] && [ "$LATEST_ITER" -gt 0 ]; then
    echo "续训: ${LATEST_EXP}  iter=${LATEST_ITER}/${TRAIN_ITERS}"
    cat > "$OVERLAY" << EOF
args:
  load: ${LATEST_EXP}
EOF
    BASE_YAMLS="${BASE_YAMLS} ${OVERLAY}"
else
    echo "从头训练: load=${PRETRAINED}"
fi

NGPUS=$(nvidia-smi -L | wc -l)
echo "=========================================="
echo " 4K LTX-2 VAE Training (2x B200, 10 epochs, 24h)"
echo " Start: $(date)"
echo " Node:  $(hostname)"
echo " GPUs:  $NGPUS"
echo " Iters: ${LATEST_ITER} → ${TRAIN_ITERS}"
echo " Save:  ${SAVE_ROOT}"
echo "=========================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000
# 默认关掉 AUDIO_DEBUG：每步双卡 print 容易把 DataLoader/GIL 卡死，导致 NCCL timeout
unset HALLO3_AUDIO_DEBUG

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TORCH_DISABLE_ADDR2LINE=1 \
torchrun --standalone --nproc_per_node=$NGPUS \
    train_video.py \
    --base $BASE_YAMLS \
    --seed $RANDOM

TRAIN_EXIT=$?

echo "=========================================="
echo " Exit code: $TRAIN_EXIT"
echo " End: $(date)"
echo "=========================================="

CKPT_INFO=$(find_latest_ckpt)
LATEST_EXP="${CKPT_INFO%%|*}"
LATEST_ITER="${CKPT_INFO##*|}"
LATEST_ITER="${LATEST_ITER:-0}"
echo "Checkpoints: ${SAVE_ROOT}"
echo "Latest: ${LATEST_EXP}  iter=${LATEST_ITER}/${TRAIN_ITERS}"
ls -lh "$SAVE_ROOT" 2>/dev/null || true

if [ "$LATEST_ITER" -ge "$TRAIN_ITERS" ]; then
    echo "10 epochs 完成，不再重提交"
    exit 0
fi

if [ "$TRAIN_EXIT" -ne 0 ]; then
    echo "训练异常退出 (exit ${TRAIN_EXIT})，不自动重提交，请检查日志"
    exit "$TRAIN_EXIT"
fi

echo "尚未到 ${TRAIN_ITERS} steps，提交下一轮"
safe_resubmit
exit 0
