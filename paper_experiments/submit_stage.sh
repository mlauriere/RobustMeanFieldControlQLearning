#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
 echo "Usage: $0 manifests/stage3_final.csv [array_concurrency]"
 exit 2
fi

MANIFEST="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
CONCURRENCY="${2:-4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N_ROWS=$(($(wc -l < "$MANIFEST") - 1))
if [[ "$N_ROWS" -le 0 ]]; then
 echo "Manifest has no data rows: $MANIFEST"
 exit 2
fi
LAST_INDEX=$((N_ROWS - 1))
mkdir -p "${SCRIPT_DIR}/logs"

sbatch --array=0-"${LAST_INDEX}"%"${CONCURRENCY}" \
 --output="${SCRIPT_DIR}/logs/%x_%A_%a.out" \
 --error="${SCRIPT_DIR}/logs/%x_%A_%a.err" \
 "${SCRIPT_DIR}/slurm_array_task.sh" "$MANIFEST"
