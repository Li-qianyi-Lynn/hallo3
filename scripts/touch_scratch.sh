#!/bin/bash
# Refresh atime AND mtime on all files under scratch to prevent 30-day purge.
# Usage: bash touch_scratch.sh [directory]
# Default directory: /scratch/li_qiany_neu

SCRATCH_DIR="${1:-/scratch/li_qiany_neu}"

echo "$(date '+%Y-%m-%d %H:%M:%S') Refreshing timestamps under: ${SCRATCH_DIR}"

# Count files and directories
file_count=$(find "${SCRATCH_DIR}" -type f | wc -l)
dir_count=$(find "${SCRATCH_DIR}" -type d | wc -l)
echo "Found: ${file_count} files, ${dir_count} directories"

# touch without -a updates both atime and mtime
find "${SCRATCH_DIR}" -exec touch {} +

echo "$(date '+%Y-%m-%d %H:%M:%S') Done. Touched ${file_count} files and ${dir_count} directories."
