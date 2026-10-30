#!/bin/bash
# Refresh atime AND mtime on all files under scratch to prevent 30-day purge.
# Usage: bash touch_scratch.sh [directory]
# Default directory: /scratch/li_qiany_neu

SCRATCH_DIR="${1:-/scratch/li_qiany_neu}"

echo "$(date '+%Y-%m-%d %H:%M:%S') Refreshing timestamps under: ${SCRATCH_DIR}"

# Count files first
total=$(find "${SCRATCH_DIR}" -type f | wc -l)
echo "Total files found: ${total}"

# touch without -a updates both atime and mtime
find "${SCRATCH_DIR}" -type f -exec touch {} +

echo "$(date '+%Y-%m-%d %H:%M:%S') Done. Touched ${total} files."
