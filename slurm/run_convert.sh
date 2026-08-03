#!/bin/bash
#SBATCH -p short
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH -o /scratch/li.qianyi/TalkVid/logs/convert_%j.log

# ==== 路径配置 ====
CLIPS_FLAT="/scratch/li.qianyi/TalkVid/clips_flat"
OUTPUT="/scratch/li.qianyi/hallo3_data"
HALLO3_DIR="/scratch/li.qianyi/hallo3"
LOG_DIR="/scratch/li.qianyi/TalkVid/logs"

mkdir -p "$OUTPUT" "$LOG_DIR"

# ==== 激活环境 ====
source /home/li.qianyi/.bashrc
conda activate /home/li.qianyi/envs/hallo

echo "开始 convert: $(date)"
echo "输入: $CLIPS_FLAT"
echo "输出: $OUTPUT"

cd "$HALLO3_DIR"

python scripts/convert_talkvid_to_hallo3.py \
    --clips_flat "$CLIPS_FLAT" \
    --output     "$OUTPUT" \
    --dataset_name talkvid \
    --num_workers 32

echo "✅ convert 完成: $(date)"

# ==== Step 4：生成训练索引 ====
echo "开始生成训练索引..."
python hallo3/extract_meta_info.py \
    -r "$OUTPUT" \
    -n talkvid

echo "✅ 训练索引生成完成: $(date)"
echo "索引文件: $HALLO3_DIR/data/talkvid.json"
