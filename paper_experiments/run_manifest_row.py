#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

from campaign_lib import (
 CAMPAIGN_DIR,
 P_HAT,
 Timer,
 compute_policy_costs,
 default_noise_grid,
 ensure_single_thread_env,
 evaluate_profile,
 lambda_grid_from_name,
 make_eval_laws,
 make_log_checkpoints,
 make_tables_for_row,
 parse_float_list,
 read_manifest_row,
 row_bool,
 row_float,
 row_int,
 run_async_qlearning_with_stats,
 run_sampled_qlearning_trace,
 run_value_iteration_trace,
 timestamp,
 write_json,
)


class Tee:
 def __init__(self, *streams):
  self.streams = streams

 def write(self, data):
  for stream in self.streams:
   stream.write(data)
   stream.flush()

 def flush(self):
  for stream in self.streams:
   stream.flush()


def run_idealized_profile(row, run_dir):
 example = row["example"]
 env, tables = make_tables_for_row(row)
 m_values = parse_float_list(row["m_grid"])
 p_values = parse_float_list(row["p_grid"])
 discount = row_float(row, "discount", 0.5)
 q_norm = row_int(row, "q_norm", 1)
 lambda_grid = lambda_grid_from_name(row.get("lambda_grid", "default"))
 max_iter = row_int(row, "idealized_max_iter", 1000)
 tol = row_float(row, "idealized_tol", 1e-9)

 q_values = []
 residuals = []
 iterations = []
 converged = []
 for m_val in m_values:
  print(f"[idealized] {example} M={m_val:g}")
  trace = run_value_iteration_trace(
   tables,
   float(m_val),
   discount,
   q_norm,
   P_HAT,
   lambda_grid,
   max_iter=max_iter,
   tol=tol,
  )
  q_values.append(trace["Q_ref"])
  residuals.append(float(trace["bellman_residual_inf"][-1]))
  iterations.append(int(trace["iterations"][-1]))
  converged.append(bool(trace["converged"]))

 q_values = np.asarray(q_values, dtype=float)
 costs = evaluate_profile(tables, q_values, p_values, m_values, example, discount)
 np.savez_compressed(
  run_dir / "results.npz",
  costs=costs,
  p_values=p_values,
  m_values=m_values,
  final_residuals=np.asarray(residuals, dtype=float),
  iterations=np.asarray(iterations, dtype=int),
  converged=np.asarray(converged, dtype=bool),
  n_disc=row_int(row, "n_disc", 0),
  a_disc=row_int(row, "a_disc", 0),
  discount=discount,
  q_norm=q_norm,
  noise_grid=default_noise_grid(example),
  p_hat=P_HAT,
 )
 if row_bool(row, "save_q", True):
  np.savez_compressed(run_dir / "q_tables.npz", m_values=m_values, q_values=q_values)
 return {
  "cost_min": float(np.min(costs)),
  "cost_max": float(np.max(costs)),
  "final_residual_max": float(np.max(residuals)),
  "iterations_max": int(np.max(iterations)),
  "all_converged": bool(all(converged)),
  "S": int(tables["S"]),
  "A": int(tables["A"]),
 }


def run_sampled_profile(row, run_dir):
 example = row["example"]
 env, tables = make_tables_for_row(row)
 m_values = parse_float_list(row["m_grid"])
 p_values = parse_float_list(row["p_grid"])
 discount = row_float(row, "discount", 0.5)
 q_norm = row_int(row, "q_norm", 1)
 lambda_grid = lambda_grid_from_name(row.get("lambda_grid", "default"))
 num_updates = row_int(row, "num_updates", 0)
 seed = row_int(row, "seed", 0)
 w_lr = row_float(row, "w_lr", 0.7)

 q_values = []
 coverage = []
 min_visits = []
 mean_visits = []
 bellman = []
 pass_counts = []
 for m_idx, m_val in enumerate(m_values):
  run_seed = int(seed) + 1000003 * m_idx
  print(f"[sampled] {example} M={m_val:g} seed={seed} run_seed={run_seed} updates={num_updates}")
  out = run_async_qlearning_with_stats(
   tables,
   float(m_val),
   discount,
   q_norm,
   P_HAT,
   lambda_grid,
   num_updates=num_updates,
   seed=run_seed,
   w_lr=w_lr,
   coverage_passes=row_int(row, "coverage_passes", 0),
   interleaved_coverage_passes=row_int(row, "interleaved_coverage_passes", 0),
   interleaved_coverage_interval=row_int(row, "interleaved_coverage_interval", 0),
  )
  q_values.append(out["Q"])
  coverage.append(out["coverage_pct"])
  min_visits.append(out["min_visits"])
  mean_visits.append(out["mean_visits"])
  bellman.append(out["bellman_residual_inf"])
  pass_counts.append(out["permutation_passes"])

 q_values = np.asarray(q_values, dtype=float)
 costs = evaluate_profile(tables, q_values, p_values, m_values, example, discount)
 np.savez_compressed(
  run_dir / "results.npz",
  costs=costs,
  p_values=p_values,
  m_values=m_values,
  coverage_pct=np.asarray(coverage, dtype=float),
  min_visits=np.asarray(min_visits, dtype=float),
  mean_visits=np.asarray(mean_visits, dtype=float),
  bellman_residual_inf=np.asarray(bellman, dtype=float),
  permutation_passes=np.asarray(pass_counts, dtype=int),
  n_disc=row_int(row, "n_disc", 0),
  a_disc=row_int(row, "a_disc", 0),
  discount=discount,
  q_norm=q_norm,
  num_updates=num_updates,
  seed=seed,
  w_lr=w_lr,
  noise_grid=default_noise_grid(example),
  p_hat=P_HAT,
 )
 if row_bool(row, "save_q", True):
  np.savez_compressed(run_dir / "q_tables.npz", m_values=m_values, q_values=q_values)
 return {
  "cost_min": float(np.min(costs)),
  "cost_max": float(np.max(costs)),
  "coverage_min": float(np.min(coverage)),
  "min_visits_min": float(np.min(min_visits)),
  "bellman_residual_max": float(np.max(bellman)),
  "S": int(tables["S"]),
  "A": int(tables["A"]),
 }


def run_lambda_sensitivity(row, run_dir):
 row_default = dict(row)
 row_dense = dict(row)
 row_default["lambda_grid"] = "default"
 row_dense["lambda_grid"] = "dense"
 tmp_default = run_dir / "default"
 tmp_dense = run_dir / "dense"
 tmp_default.mkdir(exist_ok=True)
 tmp_dense.mkdir(exist_ok=True)
 metrics_default = run_idealized_profile(row_default, tmp_default)
 metrics_dense = run_idealized_profile(row_dense, tmp_dense)
 d0 = np.load(tmp_default / "results.npz")
 d1 = np.load(tmp_dense / "results.npz")
 diff = d1["costs"] - d0["costs"]
 np.savez_compressed(
  run_dir / "results.npz",
  costs_default=d0["costs"],
  costs_dense=d1["costs"],
  cost_diff=diff,
  p_values=d0["p_values"],
  m_values=d0["m_values"],
  final_residuals_default=d0["final_residuals"],
  final_residuals_dense=d1["final_residuals"],
 )
 return {
  "max_abs_cost_diff": float(np.max(np.abs(diff))),
  "default_final_residual_max": metrics_default["final_residual_max"],
  "dense_final_residual_max": metrics_dense["final_residual_max"],
 }


def run_convergence(row, run_dir):
 example = row["example"]
 env, tables = make_tables_for_row(row)
 m_values = parse_float_list(row["m_grid"])
 if len(m_values) != 1:
  raise ValueError("convergence rows must contain exactly one M value")
 robust_m = float(m_values[0])
 discount = row_float(row, "discount", 0.5)
 q_norm = row_int(row, "q_norm", 1)
 lambda_grid = lambda_grid_from_name(row.get("lambda_grid", "default"))
 num_updates = row_int(row, "num_updates", 0)
 seed = row_int(row, "seed", 0)
 w_lr = row_float(row, "w_lr", 0.7)
 checkpoints = make_log_checkpoints(
  tables["S"] * tables["A"],
  num_updates,
  n_points=row_int(row, "n_checkpoints", 40),
 )

 print(f"[convergence] {example} M={robust_m:g} seed={seed} updates={num_updates}")
 ideal_trace = run_value_iteration_trace(
  tables,
  robust_m,
  discount,
  q_norm,
  P_HAT,
  lambda_grid,
  max_iter=row_int(row, "idealized_max_iter", 1000),
  tol=row_float(row, "idealized_tol", 1e-9),
 )
 Q_ref = ideal_trace["Q_ref"]
 eval_laws = make_eval_laws(example)
 ref_costs = compute_policy_costs(tables, Q_ref, eval_laws, discount)
 sampled = run_sampled_qlearning_trace(
  tables,
  robust_m,
  discount,
  q_norm,
  P_HAT,
  lambda_grid,
  num_updates=num_updates,
  checkpoints=checkpoints,
  seed=seed,
  Q_ref=Q_ref,
  ref_costs=ref_costs,
  eval_laws=eval_laws,
  w_lr=w_lr,
 )

 fieldnames = [
  "step",
  "q_sup_error",
  "q_mae_error",
  "bellman_residual_inf",
  "coverage_pct",
  "min_visits",
  "mean_visits",
  "policy_gap_abs_p0",
  "policy_gap_abs_p05",
  "policy_gap_abs_p1",
  "policy_cost_p1",
  "reference_cost_p1",
 ]
 with (run_dir / "convergence_trace.csv").open("w", newline="") as handle:
  writer = csv.DictWriter(handle, fieldnames=fieldnames)
  writer.writeheader()
  for record in sampled["records"]:
   out = {field: record.get(field, "") for field in fieldnames}
   out["step"] = record["updates"]
   writer.writerow(out)

 np.savez_compressed(
  run_dir / "results.npz",
  checkpoints=checkpoints,
  idealized_iterations=ideal_trace["iterations"],
  idealized_residuals=ideal_trace["bellman_residual_inf"],
  idealized_q_sup_errors=ideal_trace["q_sup_error"],
  sampled_records=np.asarray(
   [
    [
     r["updates"],
     r["q_sup_error"],
     r["q_mae_error"],
     r["bellman_residual_inf"],
     r["coverage_pct"],
     r["min_visits"],
     r["policy_gap_abs_p1"],
    ]
    for r in sampled["records"]
   ],
   dtype=float,
  ),
  m_values=np.asarray([robust_m], dtype=float),
  discount=discount,
  seed=seed,
  w_lr=w_lr,
 )
 if row_bool(row, "save_q", True):
  np.savez_compressed(
   run_dir / "q_tables.npz",
   m_values=np.asarray([robust_m], dtype=float),
   q_values=np.asarray([sampled["Q"]], dtype=float),
   q_ref=np.asarray([Q_ref], dtype=float),
  )
 final = sampled["records"][-1]
 return {
  "ideal_final_residual": float(ideal_trace["bellman_residual_inf"][-1]),
  "q_sup_error_final": float(final["q_sup_error"]),
  "policy_gap_abs_p1_final": float(final["policy_gap_abs_p1"]),
  "coverage_final": float(final["coverage_pct"]),
  "min_visits_final": float(final["min_visits"]),
  "S": int(tables["S"]),
  "A": int(tables["A"]),
 }


TASKS = {
 "idealized_profile": run_idealized_profile,
 "sampled_profile": run_sampled_profile,
 "lambda_sensitivity": run_lambda_sensitivity,
 "convergence": run_convergence,
}


def row_mismatches(existing, current):
 fields = sorted(set(existing) | set(current))
 return [
  field for field in fields
  if str(existing.get(field, "")) != str(current.get(field, ""))
 ]


def run_row(row, campaign_root, force=False):
 run_dir = Path(campaign_root) / "runs" / row["stage"] / row["run_id"]
 done_path = run_dir / "DONE.json"
 failed_path = run_dir / "FAILED.json"
 if done_path.exists() and not force:
  row_path = run_dir / "row.json"
  if row_path.exists():
   with row_path.open() as handle:
    existing_row = json.load(handle)
   mismatches = row_mismatches(existing_row, row)
   if mismatches:
    preview = ", ".join(mismatches[:8])
    raise RuntimeError(
     f"Completed row {row['run_id']} does not match the current manifest "
     f"({preview}). Use --force only if this rerun is intentional."
    )
  print(f"Skipping completed row {row['run_id']} ({done_path})")
  return
 run_dir.mkdir(parents=True, exist_ok=True)
 for marker in (done_path, failed_path):
  if marker.exists():
   marker.unlink()
 write_json(run_dir / "row.json", row)
 log_handle = (run_dir / "run.log").open("w")
 tee_out = Tee(sys.__stdout__, log_handle)
 tee_err = Tee(sys.__stderr__, log_handle)

 timer = Timer()
 started_at = timestamp()
 try:
  with redirect_stdout(tee_out), redirect_stderr(tee_err):
   print(f"=== RUN {row['run_id']} | {row['task']} | {row['example']} ===")
   print(json.dumps(row, indent=2, sort_keys=True))
   task = TASKS[row["task"]]
   task_metrics = task(row, run_dir)
   metrics = {
    "run_id": row["run_id"],
    "stage": row["stage"],
    "task": row["task"],
    "example": row["example"],
    "started_at": started_at,
    "finished_at": timestamp(),
    "runtime_seconds": timer.seconds(),
    "status": "done",
    "thread_env": {
     "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", ""),
     "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
     "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
    },
   }
   metrics.update(task_metrics)
   write_json(run_dir / "metrics.json", metrics)
   write_json(done_path, metrics)
   print(f"=== DONE {row['run_id']} in {metrics['runtime_seconds']:.1f}s ===")
 except Exception as exc:
  failed = {
   "run_id": row.get("run_id", ""),
   "stage": row.get("stage", ""),
   "task": row.get("task", ""),
   "example": row.get("example", ""),
   "started_at": started_at,
   "failed_at": timestamp(),
   "runtime_seconds": timer.seconds(),
   "status": "failed",
   "error": repr(exc),
   "traceback": traceback.format_exc(),
  }
  write_json(failed_path, failed)
  raise
 finally:
  log_handle.close()


def main():
 parser = argparse.ArgumentParser()
 parser.add_argument("--manifest", required=True)
 parser.add_argument("--row-index", type=int, default=-1)
 parser.add_argument("--campaign-root", type=str, default=str(CAMPAIGN_DIR))
 parser.add_argument("--force", action="store_true")
 args = parser.parse_args()

 ensure_single_thread_env()
 row_index = args.row_index
 if row_index < 0:
  if "SLURM_ARRAY_TASK_ID" not in os.environ:
   raise ValueError("--row-index is required outside Slurm array jobs")
  row_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
 row, total = read_manifest_row(args.manifest, row_index)
 print(f"Loaded manifest row {row_index}/{total - 1}: {row['run_id']}")
 run_row(row, Path(args.campaign_root).resolve(), force=args.force)


if __name__ == "__main__":
 main()
