#!/bin/bash
# Refresh access time on all files under scratch to prevent 30-day purge.
# Usage: bash touch_scratch.sh [directory]
# Default directory: /scratch/li_qiany_neu

SCRATCH_DIR="${1:-/scratch/li_qiany_neu}"

echo "$(date '+%Y-%m-%d %H:%M:%S') Refreshing access times under: ${SCRATCH_DIR}"

count=0
find "${SCRATCH_DIR}" -type f -print0 | while IFS= read -r -d '' file; do
    touch -a "$file" 2>/dev/null
    count=$((count + 1))
    if (( count % 10000 == 0 )); then
        echo "  ... touched ${count} files"
    fi
done

echo "$(date '+%Y-%m-%d %H:%M:%S') Done."
