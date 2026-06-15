#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${ROBUST_MFC_PYTHON:-python3}"
MAX_CONCURRENT="${ROBUST_MFC_ARRAY_THROTTLE:-}"
STAGE3_SEIR_THRESHOLD="${ROBUST_MFC_SEIR_ESCALATION_THRESHOLD:-0.01}"
POLL_SECONDS="${ROBUST_MFC_POLL_SECONDS:-120}"

cd "$ROOT"
mkdir -p logs

stamp="$(date +%Y%m%d-%H%M%S)"
ledger="logs/autonomous_pipeline_${stamp}.log"

log() {
 echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$ledger"
}

array_spec() {
 local first="$1"
 local last="$2"
 if [[ -n "$MAX_CONCURRENT" && "$MAX_CONCURRENT" != "0" ]]; then
 echo "${first}-${last}%${MAX_CONCURRENT}"
 else
 echo "${first}-${last}"
 fi
}

submit_stage() {
 local stage_name="$1"
 local manifest="$2"
 local first="$3"
 local last="$4"
 local array
 array="$(array_spec "$first" "$last")"
 log "Submitting ${stage_name}: manifest=${manifest}, array=${array}"
 sbatch --parsable \
 --job-name="${stage_name}" \
 --time=12:00:00 \
 --cpus-per-task=1 \
 --mem=4G \
 --array="${array}" \
 --export=ALL,PY="$PY",MANIFEST="$ROOT/$manifest",ROOT="$ROOT" \
 --output="$ROOT/logs/%x_%A_%a.out" \
 --error="$ROOT/logs/%x_%A_%a.err" \
 --wrap='export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; cd "$ROOT"; "$PY" run_manifest_row.py --manifest "$MANIFEST" --row-index "$SLURM_ARRAY_TASK_ID" --campaign-root "$ROOT"'
}

wait_for_job() {
 local job_id="$1"
 log "Waiting for job ${job_id}"
 while squeue -h -j "$job_id" | grep -q .; do
 squeue -j "$job_id" -o "%.18i %.20j %.8T %.10M %.10l %R" | tee -a "$ledger" || true
 sleep "$POLL_SECONDS"
 done
 log "Job ${job_id} left the queue"
 sacct -j "$job_id" --format=JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS | tee -a "$ledger" || true
}

collect() {
 log "Collecting results"
 "$PY" collect_hpc_results.py | tee -a "$ledger"
 cat aggregates/manifest_status.csv | tee -a "$ledger"
 cat aggregates/validation_report.json | tee -a "$ledger"
}

check_no_failed_rows() {
 "$PY" - <<'PY'
import json
p = json.load(open("aggregates/validation_report.json"))
if p.get("failed_runs", 0) or p.get("error_count", 0):
 raise SystemExit(f"validation failed: failed_runs={p.get('failed_runs')} error_count={p.get('error_count')}")
PY
}

rebuild_from_recommendation() {
 if [[ ! -f aggregates/systemic_hyperparameter_recommendation.json ]]; then
 log "No Systemic Risk recommendation file found; leaving manifests unchanged"
 return
 fi
 log "Rebuilding downstream manifests from Systemic Risk recommendation"
 read -r selected_w selected_updates < <("$PY" - <<'PY'
import json
p = json.load(open("aggregates/systemic_hyperparameter_recommendation.json"))
s = p["selected"]
print(s["w_lr"], int(s["num_updates"]))
PY
)
 log "Selected Systemic Risk setting: w=${selected_w}, updates=${selected_updates}"
 "$PY" build_hpc_manifests.py --systemic-w "$selected_w" --systemic-updates "$selected_updates" | tee -a "$ledger"
}

seir_needs_escalation() {
 "$PY" - "$STAGE3_SEIR_THRESHOLD" <<'PY'
import csv
import sys
threshold = float(sys.argv[1])
vals = []
with open("aggregates/sampled_vs_idealized_gaps.csv", newline="") as handle:
 for row in csv.DictReader(handle):
  if row["stage"] == "stage3_final" and row["example"] == "seir":
   vals.append(float(row["abs_gap"]))
if not vals:
 print("no_seir_stage3_rows")
 raise SystemExit(2)
mean_gap = sum(vals) / len(vals)
max_gap = max(vals)
print(f"SEIR stage3 mean_abs_gap={mean_gap:.12g} max_abs_gap={max_gap:.12g} threshold={threshold:.12g}")
raise SystemExit(0 if max_gap > threshold else 1)
PY
}

log "Autonomous pipeline started in $ROOT"
log "Python executable: $PY"
log "Array throttle: ${MAX_CONCURRENT:-none}"

collect
check_no_failed_rows
rebuild_from_recommendation

JOB3="$(submit_stage run_stage3 manifests/stage3_final.csv 0 59 | tail -n 1)"
log "Stage 3 job id: $JOB3"
wait_for_job "$JOB3"
collect
check_no_failed_rows

if seir_needs_escalation | tee -a "$ledger"; then
 log "Submitting optional SEIR escalation"
 JOB3B="$(submit_stage run_stage3b manifests/stage3_seir_escalation.csv 0 9 | tail -n 1)"
 log "Stage 3b job id: $JOB3B"
 wait_for_job "$JOB3B"
 collect
 check_no_failed_rows
else
 log "Skipping optional SEIR escalation"
fi

JOB4="$(submit_stage run_stage4 manifests/stage4_convergence.csv 0 119 | tail -n 1)"
log "Stage 4 job id: $JOB4"
wait_for_job "$JOB4"
collect
check_no_failed_rows

JOB5="$(submit_stage run_stage5 manifests/stage5_grid_sensitivity.csv 0 25 | tail -n 1)"
log "Stage 5 job id: $JOB5"
wait_for_job "$JOB5"
collect
check_no_failed_rows

log "Autonomous pipeline complete"
