#!/bin/bash
#SBATCH --job-name=train-g1-2gpu
#SBATCH --partition=rtx-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --account=p2026_0014_neu
#SBATCH --output=/scratch/li_qiany_neu/hallo3_logs/train_g1_%j.log
#
# G1: Omnilingual wav2vec2 + adapter (2×GPU, 7000 steps)
#   2700 clips / 2 GPU = 1350 steps/epoch
#   ~5.2 epochs ≈ 7000 steps ≈ ~9h @ ~9s/step
#   超时前 3 分钟自动 sbatch 下一轮
# 提交: sbatch slurm/run_train_g1.sh

REPO_DIR="${HOME}/hallo3"
THIS_SCRIPT="${REPO_DIR}/slurm/run_train_g1.sh"
SAVE_ROOT="/scratch/li_qiany_neu/train_output_g1"
PRETRAINED="/scratch/li_qiany_neu/pretrained_models/hallo3"
LOG_DIR="/scratch/li_qiany_neu/hallo3_logs"
TRAIN_ITERS=7000
JOB_NAME="train-g1-2gpu"

mkdir -p "$LOG_DIR" "$SAVE_ROOT"

# ---- 超时前自动重提交（USR1，不杀当前训练）----
RESUBMITTED=0
safe_resubmit() {
    if [ "$RESUBMITTED" = "1" ]; then
        return
    fi
    if [ "${NO_RESUBMIT:-0}" = "1" ]; then
        echo "[resubmit] NO_RESUBMIT=1，跳过"
        return
    fi
    already=$(squeue -u "$USER" -h -n "$JOB_NAME" -o "%i" 2>/dev/null \
        | grep -v "^${SLURM_JOB_ID}$" | wc -l | tr -d ' ')
    if [ "${already:-0}" -ge 1 ]; then
        echo "[resubmit] 队列中已有 ${JOB_NAME}，跳过"
        return
    fi
    echo "[resubmit] sbatch ${THIS_SCRIPT}  $(date)"
    sbatch --dependency=afterany:${SLURM_JOB_ID} "$THIS_SCRIPT"
    RESUBMITTED=1
}

on_time_warning() {
    echo "=========================================="
    echo " USR1: 距超时约 3 分钟，排队下一轮  $(date)"
    echo "=========================================="
    safe_resubmit
}
trap on_time_warning USR1

# ---- 汇总 checkpoint 进度 ----
scan_ckpts() {
    TOTAL_DONE=0
    LOAD_DIR=""
    local f iter m newest=0
    while IFS= read -r f; do
        [ -f "$f" ] || continue
        iter=$(tr -d '[:space:]' < "$f")
        [[ "$iter" =~ ^[0-9]+$ ]] || continue
        [ "$iter" -gt 0 ] || continue
        TOTAL_DONE=$((TOTAL_DONE + iter))
        m=$(stat -c %Y "$f" 2>/dev/null || stat -f %m "$f")
        if [ "$m" -ge "$newest" ]; then
            newest=$m
            LOAD_DIR=$(dirname "$f")
        fi
    done < <(find "$SAVE_ROOT" -name latest -type f 2>/dev/null)
}

scan_ckpts
REMAINING=$((TRAIN_ITERS - TOTAL_DONE))

if [ "$REMAINING" -le 0 ]; then
    echo "训练已完成: 累计 ${TOTAL_DONE}/${TRAIN_ITERS}"
    exit 0
fi

module load miniforge3
module load cuda
source activate hallo3

cd "${REPO_DIR}/hallo3"
ln -sf /scratch/li_qiany_neu/pretrained_models pretrained_models

OVERLAY="${LOG_DIR}/sft_g1_overlay_${SLURM_JOB_ID}.yaml"
BASE_YAMLS="../configs/cogvideox_5b_i2v_s2.yaml ../configs/sft_g1_train.yaml"
if [ -n "$LOAD_DIR" ]; then
    echo "续训: load=${LOAD_DIR}  累计 ${TOTAL_DONE}/${TRAIN_ITERS}，本轮目标 ${REMAINING} 步"
    cat > "$OVERLAY" << EOF
args:
  load: ${LOAD_DIR}
  train_iters: ${REMAINING}
EOF
    BASE_YAMLS="${BASE_YAMLS} ${OVERLAY}"
else
    echo "从头训练: load=${PRETRAINED}，本轮目标 ${REMAINING} 步"
fi

NGPUS=$(nvidia-smi -L | wc -l)
if [ "$NGPUS" -ne 2 ]; then
    echo "期望本节点 2 张 GPU，实际 ${NGPUS}（节点: ${SLURM_JOB_NODELIST}），退出"
    exit 2
fi
echo "=========================================="
echo " G1: Omnilingual + Adapter Training (2x GPU, 7000 steps)"
echo " Start: $(date)"
echo " Node:  $(hostname)"
echo " GPUs:  $NGPUS"
echo " Iters: 累计 ${TOTAL_DONE}/${TRAIN_ITERS}，本轮 ${REMAINING}"
echo " Save:  ${SAVE_ROOT}"
echo "=========================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
unset HALLO3_AUDIO_DEBUG

# wandb: WANDB=1 sbatch slurm/run_train_g1.sh
WANDB_ARGS=""
if [ "${WANDB:-0}" = "1" ]; then
    WANDB_ARGS="--wandb"
    export WANDB_DIR="${LOG_DIR}"
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    echo "wandb: 开启 (WANDB_MODE=${WANDB_MODE:-online}, WANDB_DIR=${WANDB_DIR})"
fi

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TORCH_DISABLE_ADDR2LINE=1 \
torchrun --standalone --nproc_per_node=$NGPUS \
    train_video.py \
    --base $BASE_YAMLS \
    --seed $RANDOM \
    $WANDB_ARGS &
TRAIN_PID=$!

TRAIN_EXIT=0
while true; do
    wait "$TRAIN_PID"
    rc=$?
    if kill -0 "$TRAIN_PID" 2>/dev/null; then
        continue
    fi
    TRAIN_EXIT=$rc
    break
done

echo "=========================================="
echo " Exit code: $TRAIN_EXIT"
echo " End: $(date)"
echo "=========================================="

scan_ckpts
echo "Checkpoints: ${SAVE_ROOT}"
echo "最新目录: ${LOAD_DIR}  累计 ${TOTAL_DONE}/${TRAIN_ITERS}"
ls -lh "$SAVE_ROOT" 2>/dev/null || true

if [ "$TOTAL_DONE" -ge "$TRAIN_ITERS" ]; then
    echo "7000 steps 完成，不再重提交"
    exit 0
fi

if [ "$TRAIN_EXIT" -ne 0 ]; then
    echo "训练异常退出 (exit ${TRAIN_EXIT})，不自动重提交，请检查日志"
    exit "$TRAIN_EXIT"
fi

echo "尚未到 ${TRAIN_ITERS} steps，提交下一轮"
safe_resubmit
exit 0
