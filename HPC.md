# HPC Instructions

Full reproduction of the sampled experiments requires an HPC cluster. The
aggregate results included in `paper_results/` were produced by the HPC
campaign described here.

## Prerequisites

- Slurm workload manager with array job support.
- Python environment with NumPy, SciPy, Matplotlib.
- Single-core task allocation (each row runs on one CPU core).

## Environment Setup

```bash
export ROBUST_MFC_PYTHON=/path/to/python
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## Campaign Structure

Each campaign is defined by a manifest CSV file. Each row specifies one
configuration (environment, grid size, robustness radius, seed, etc.).

### Key Files in `paper_experiments/`

- `build_hpc_manifests.py` — Generates manifest CSV files from experiment
  parameters.
- `run_manifest_row.py` — Runs a single manifest row.
- `slurm_array_task.sh` — Slurm task template for array jobs.
- `submit_stage.sh` — Submits a campaign stage as an array job. Uses
  zero-based array indexing (0..N_ROWS-1), consistent with
  `$SLURM_ARRAY_TASK_ID` usage in `slurm_array_task.sh`.
- `collect_hpc_results.py` — Aggregates completed runs, generates figures and
  tables.

### Running a Campaign

1. Build the manifest (run from repo root):
   ```bash
   python paper_experiments/build_hpc_manifests.py
   ```
   This writes CSV files to `paper_experiments/manifests/`.

2. Submit array jobs from the `paper_experiments/` directory:
   ```bash
   cd paper_experiments
   bash submit_stage.sh manifests/stage3_final.csv
   ```

   `submit_stage.sh` computes the number of rows, sets zero-based array
   indices (0..N_ROWS-1), and invokes `sbatch` with `slurm_array_task.sh`.
   Optionally pass a concurrency limit:
   ```bash
   bash submit_stage.sh manifests/stage3_final.csv 4
   ```

3. After all rows complete, aggregate results:
   ```bash
   python paper_experiments/collect_hpc_results.py
   ```

## HPC Path Conventions

The campaign scripts use relative paths from the repository root.
`slurm_array_task.sh` sets the working directory to the repo root before
invoking `run_manifest_row.py`.  Set `ROBUST_MFC_PYTHON` to the Python
interpreter path before submitting.

Example deployment:

```bash
# Set paths
export WORK_DIR=/path/to/repo
export ROBUST_MFC_PYTHON=/path/to/python

# Build manifests and submit
cd $WORK_DIR
python paper_experiments/build_hpc_manifests.py
cd paper_experiments
bash submit_stage.sh manifests/stage3_final.csv
```
