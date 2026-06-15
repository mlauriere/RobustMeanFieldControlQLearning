#!/usr/bin/env bash
#SBATCH --job-name=robust_mfc
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$1"
FORCE_ARGS=()
if [[ "${ROBUST_MFC_FORCE:-0}" != "0" ]]; then
 FORCE_ARGS+=(--force)
fi

if [[ -n "${ROBUST_MFC_MODULE:-}" ]]; then
 module load "${ROBUST_MFC_MODULE}"
fi
if [[ -n "${ROBUST_MFC_CONDA_ENV:-}" ]]; then
 eval "$(conda shell.bash hook)"
 conda activate "${ROBUST_MFC_CONDA_ENV}"
fi

cd "${SCRIPT_DIR}/.."
"${ROBUST_MFC_PYTHON:-python3}" "${SCRIPT_DIR}/run_manifest_row.py" \
 --manifest "${MANIFEST}" \
 --row-index "${SLURM_ARRAY_TASK_ID}" \
 --campaign-root "${SCRIPT_DIR}" \
 "${FORCE_ARGS[@]}"
