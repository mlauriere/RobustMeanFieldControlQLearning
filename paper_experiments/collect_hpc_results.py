#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

_CACHE_DIR = Path(__file__).resolve().parent / ".cache"
_CACHE_DIR.mkdir(exist_ok=True)
(_CACHE_DIR / "matplotlib").mkdir(exist_ok=True)
os.environ["XDG_CACHE_HOME"] = str(_CACHE_DIR)
os.environ["MPLCONFIGDIR"] = str(_CACHE_DIR / "matplotlib")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from campaign_lib import (
 CAMPAIGN_DIR,
 MANIFEST_FIELDS,
 P_FINE,
 adverse_law,
 bootstrap_ci,
 default_noise_grid,
 example_label,
 interpolate_to_grid,
 make_tables_for_row,
 make_true_law,
 policy_margin_and_disagreement,
 write_csv,
 write_json,
)


ENV_ORDER = ["sysrisk", "sis", "seir"]
COLORS = {
 0.0: "#1b9e77",
 0.005: "#4daf4a",
 0.01: "#66a61e",
 0.02: "#a6d854",
 0.03: "#b3de69",
 0.04: "#ccebc5",
 0.05: "#d95f02",
 0.06: "#fdb462",
 0.08: "#fb8072",
 0.1: "#7570b3",
 0.15: "#bc80bd",
 0.2: "#fccde5",
 0.3: "#e7298a",
 0.4: "#e6ab02",
 0.5: "#a6761d",
 0.6: "#66a61e",
 0.75: "#666666",
 1.0: "#1f78b4",
}


def read_json(path):
 with Path(path).open() as handle:
  return json.load(handle)


def read_manifest_rows(root):
 manifest_dir = root / "manifests"
 rows = []
 for path in sorted(manifest_dir.glob("stage*.csv")):
  with path.open(newline="") as handle:
   for row in csv.DictReader(handle):
    item = dict(row)
    item["manifest"] = path.name
    rows.append(item)
 return rows


def discover_runs(root):
 runs = []
 for done in sorted((root / "runs").glob("*/*/DONE.json")):
  run_dir = done.parent
  row = read_json(run_dir / "row.json")
  metrics = read_json(run_dir / "metrics.json")
  runs.append({"run_dir": run_dir, "row": row, "metrics": metrics})
 return runs


def discover_failures(root):
 failures = []
 for failed in sorted((root / "runs").glob("*/*/FAILED.json")):
  payload = read_json(failed)
  payload["run_dir"] = str(failed.parent)
  failures.append(payload)
 return failures


def load_npz(run):
 path = run["run_dir"] / "results.npz"
 if not path.exists():
  return None
 return np.load(path, allow_pickle=True)


def load_q(run):
 path = run["run_dir"] / "q_tables.npz"
 if not path.exists():
  return None
 return np.load(path, allow_pickle=True)


def finite_or_nan(value):
 value = float(value)
 return value if np.isfinite(value) else np.nan


def profile_rows(runs):
 rows = []
 for run in runs:
  row = run["row"]
  if row["task"] not in {"idealized_profile", "sampled_profile"}:
   continue
  data = load_npz(run)
  if data is None:
   continue
  costs = np.asarray(data["costs"], dtype=float)
  p_values = np.asarray(data["p_values"], dtype=float)
  m_values = np.asarray(data["m_values"], dtype=float)
  for p_idx, p_val in enumerate(p_values):
   for m_idx, m_val in enumerate(m_values):
    rows.append(
     {
      "run_id": row["run_id"],
      "stage": row["stage"],
      "task": row["task"],
      "solver": "idealized" if row["task"] == "idealized_profile" else "sampled",
      "example": row["example"],
      "n_disc": row["n_disc"],
      "a_disc": row["a_disc"],
      "discount": row["discount"],
      "seed": row.get("seed", ""),
      "w_lr": row.get("w_lr", ""),
      "num_updates": row.get("num_updates", ""),
      "p": f"{p_val:.12g}",
      "M": f"{m_val:.12g}",
      "cost": f"{costs[p_idx, m_idx]:.12g}",
     }
    )
 return rows


def reference_runs(runs):
 refs = {}
 for run in runs:
  row = run["row"]
  if row["task"] != "idealized_profile":
   continue
  data = load_npz(run)
  if data is None:
   continue
  key = (row["example"], int(row["n_disc"]), int(row["a_disc"]), float(row["discount"]))
  priority = 0 if row["stage"] == "stage1_idealized" else 1
  old = refs.get(key)
  if old is None or priority < old["priority"]:
   refs[key] = {"run": run, "data": data, "priority": priority}
 return refs


def sampled_idealized_gaps(runs, refs):
 rows = []
 for run in runs:
  row = run["row"]
  if row["task"] != "sampled_profile":
   continue
  data = load_npz(run)
  if data is None:
   continue
  key = (row["example"], int(row["n_disc"]), int(row["a_disc"]), float(row["discount"]))
  ref = refs.get(key)
  if ref is None:
   continue
  sampled = np.asarray(data["costs"], dtype=float)
  p_values = np.asarray(data["p_values"], dtype=float)
  m_values = np.asarray(data["m_values"], dtype=float)
  ideal = interpolate_to_grid(
   ref["data"]["p_values"],
   ref["data"]["m_values"],
   ref["data"]["costs"],
   p_values,
   m_values,
  )
  for p_idx, p_val in enumerate(p_values):
   for m_idx, m_val in enumerate(m_values):
    gap = sampled[p_idx, m_idx] - ideal[p_idx, m_idx]
    rows.append(
     {
      "run_id": row["run_id"],
      "stage": row["stage"],
      "example": row["example"],
      "n_disc": row["n_disc"],
      "a_disc": row["a_disc"],
      "discount": row["discount"],
      "seed": row.get("seed", ""),
      "w_lr": row.get("w_lr", ""),
      "num_updates": row.get("num_updates", ""),
      "p": f"{p_val:.12g}",
      "M": f"{m_val:.12g}",
      "sampled_cost": f"{sampled[p_idx, m_idx]:.12g}",
      "idealized_cost": f"{ideal[p_idx, m_idx]:.12g}",
      "gap": f"{gap:.12g}",
      "abs_gap": f"{abs(gap):.12g}",
      "rel_gap": f"{abs(gap) / max(abs(ideal[p_idx, m_idx]), 1e-12):.12g}",
      "reference_run_id": ref["run"]["row"]["run_id"],
     }
    )
 return rows


def group_float_rows(rows, keys, value_field):
 grouped = defaultdict(list)
 for row in rows:
  key = tuple(row[k] for k in keys)
  grouped[key].append(float(row[value_field]))
 return grouped


def best_radius_rows(policy_rows):
 final_rows = [r for r in policy_rows if r["stage"] == "stage3_final" and r["solver"] == "sampled"]
 if not final_rows:
  final_rows = [r for r in policy_rows if r["solver"] == "sampled"]
 grouped = group_float_rows(final_rows, ["example", "p", "M"], "cost")
 by_example_p = defaultdict(list)
 for (example, p_val, m_val), values in grouped.items():
  mean, lo, hi = bootstrap_ci(values)
  by_example_p[(example, p_val)].append((float(m_val), mean, lo, hi, len(values)))
 rows = []
 for (example, p_val), items in sorted(by_example_p.items()):
  m0 = [item for item in items if abs(item[0]) < 1e-12]
  baseline = m0[0][1] if m0 else np.nan
  best = min(items, key=lambda x: x[1])
  rows.append(
   {
    "example": example,
    "p": p_val,
    "best_M": f"{best[0]:.12g}",
    "best_cost_mean": f"{best[1]:.12g}",
    "best_cost_ci_low": f"{best[2]:.12g}",
    "best_cost_ci_high": f"{best[3]:.12g}",
    "M0_cost_mean": f"{baseline:.12g}",
    "relative_improvement_vs_M0": f"{(baseline - best[1]) / baseline:.12g}" if np.isfinite(baseline) and baseline > 0 else "",
    "n_seeds": best[4],
   }
  )
 return rows


def relative_improvement_rows(policy_rows):
 final_rows = [r for r in policy_rows if r["stage"] == "stage3_final" and r["solver"] == "sampled"]
 if not final_rows:
  final_rows = [r for r in policy_rows if r["solver"] == "sampled"]
 grouped = group_float_rows(final_rows, ["example", "p", "M"], "cost")
 baselines = {}
 for (example, p_val, m_val), values in grouped.items():
  if abs(float(m_val)) < 1e-12:
   baselines[(example, p_val)] = np.mean(values)
 rows = []
 for (example, p_val, m_val), values in sorted(grouped.items()):
  baseline = baselines.get((example, p_val), np.nan)
  improvements = [(baseline - value) / baseline for value in values] if np.isfinite(baseline) and baseline > 0 else [np.nan]
  mean, lo, hi = bootstrap_ci(improvements)
  rows.append(
   {
    "example": example,
    "p": p_val,
    "M": m_val,
    "relative_improvement_mean": f"{mean:.12g}",
    "relative_improvement_ci_low": f"{lo:.12g}",
    "relative_improvement_ci_high": f"{hi:.12g}",
    "n_seeds": len(values),
   }
  )
 return rows


def hyperparameter_rows(gap_rows):
 rows = []
 screen = [r for r in gap_rows if r["stage"] == "stage2_sysrisk_screen" and r["example"] == "sysrisk"]
 grouped = defaultdict(list)
 for row in screen:
  if abs(float(row["p"]) - 1.0) > 1e-12:
   continue
  grouped[(row["w_lr"], row["num_updates"], row["M"])].append(float(row["abs_gap"]))
 for (w_lr, updates, m_val), values in sorted(grouped.items(), key=lambda x: (float(x[0][0]), int(float(x[0][1])), float(x[0][2]))):
  rows.append(
   {
    "example": "sysrisk",
    "w_lr": w_lr,
    "num_updates": updates,
    "M": m_val,
    "median_abs_gap_p1": f"{np.median(values):.12g}",
    "mean_abs_gap_p1": f"{np.mean(values):.12g}",
    "n_seeds": len(values),
   }
  )
 return rows


def gap_summary_rows(gap_rows):
 grouped = defaultdict(list)
 for row in gap_rows:
  grouped[(row["stage"], row["example"])].append(float(row["abs_gap"]))
 rows = []
 for (stage, example), values in sorted(grouped.items()):
  arr = np.asarray(values, dtype=float)
  rows.append(
   {
    "stage": stage,
    "example": example,
    "n": len(arr),
    "mean_abs_gap": f"{float(np.mean(arr)):.12g}",
    "max_abs_gap": f"{float(np.max(arr)):.12g}",
    "q90_abs_gap": f"{float(np.quantile(arr, 0.90)):.12g}",
    "q95_abs_gap": f"{float(np.quantile(arr, 0.95)):.12g}",
    "q99_abs_gap": f"{float(np.quantile(arr, 0.99)):.12g}",
   }
  )
 return rows


def gap_outlier_rows(gap_rows, threshold=0.01):
 rows = []
 for row in gap_rows:
  if float(row["abs_gap"]) <= threshold:
   continue
  out = dict(row)
  out["outlier_threshold"] = f"{threshold:.12g}"
  rows.append(out)
 rows.sort(key=lambda r: (r["stage"], r["example"], -float(r["abs_gap"])))
 return rows


def convergence_rows(runs):
 rows = []
 for run in runs:
  row = run["row"]
  if row["task"] != "convergence":
   continue
  path = run["run_dir"] / "convergence_trace.csv"
  if not path.exists():
   continue
  with path.open(newline="") as handle:
   for record in csv.DictReader(handle):
    out = {
     "run_id": row["run_id"],
     "stage": row["stage"],
     "study": row.get("notes", ""),
     "example": row["example"],
     "n_disc": row["n_disc"],
     "a_disc": row["a_disc"],
     "discount": row["discount"],
     "M": row["m_grid"],
     "seed": row["seed"],
     "w_lr": row["w_lr"],
     "num_updates": row["num_updates"],
    }
    out.update(record)
    rows.append(out)
 return rows


def policy_diagnostics_rows(runs, refs):
 rows = []
 table_cache = {}
 q_ref_cache = {}
 for run in runs:
  row = run["row"]
  if row["task"] != "sampled_profile" or row["stage"] not in {"stage3_final", "stage5_grid_sensitivity"}:
   continue
  q_data = load_q(run)
  if q_data is None:
   continue
  key = (row["example"], int(row["n_disc"]), int(row["a_disc"]), float(row["discount"]))
  ref = refs.get(key)
  if ref is None:
   continue
  ref_q = load_q(ref["run"])
  if ref_q is None:
   continue
  if key not in table_cache:
   _, table_cache[key] = make_tables_for_row(row)
  tables = table_cache[key]
  ref_m = np.asarray(ref_q["m_values"], dtype=float)
  ref_q_values = np.asarray(ref_q["q_values"], dtype=float)
  sampled_m = np.asarray(q_data["m_values"], dtype=float)
  sampled_q_values = np.asarray(q_data["q_values"], dtype=float)
  p_true = make_true_law(row["example"], 1.0)
  for m_idx, m_val in enumerate(sampled_m):
   ref_idx = int(np.argmin(np.abs(ref_m - m_val)))
   if abs(ref_m[ref_idx] - m_val) > 1e-8:
    continue
   diag = policy_margin_and_disagreement(
    tables,
    ref_q_values[ref_idx],
    sampled_q_values[m_idx],
    p_true,
    float(row["discount"]),
   )
   out = {
    "run_id": row["run_id"],
    "stage": row["stage"],
    "example": row["example"],
    "n_disc": row["n_disc"],
    "a_disc": row["a_disc"],
    "discount": row["discount"],
    "seed": row.get("seed", ""),
    "M": f"{m_val:.12g}",
    "reference_run_id": ref["run"]["row"]["run_id"],
   }
   out.update({k: f"{v:.12g}" for k, v in diag.items()})
   rows.append(out)
 return rows


def grid_sensitivity_rows(policy_rows):
 ideal = [r for r in policy_rows if r["solver"] == "idealized" and r["stage"] in {"stage1_idealized", "stage5_grid_sensitivity"}]
 grouped = group_float_rows(ideal, ["example", "n_disc", "a_disc", "p", "M"], "cost")
 by_grid_p = defaultdict(list)
 for (example, n_disc, a_disc, p_val, m_val), values in grouped.items():
  by_grid_p[(example, n_disc, a_disc, p_val)].append((float(m_val), float(np.mean(values))))
 rows = []
 for key, items in sorted(by_grid_p.items()):
  example, n_disc, a_disc, p_val = key
  best = min(items, key=lambda x: x[1])
  rows.append(
   {
    "example": example,
    "n_disc": n_disc,
    "a_disc": a_disc,
    "p": p_val,
    "best_M": f"{best[0]:.12g}",
    "best_cost": f"{best[1]:.12g}",
   }
  )
 return rows


def runtime_rows(runs):
 rows = []
 for run in runs:
  row = dict(run["row"])
  metrics = run["metrics"]
  rows.append(
   {
    "run_id": row["run_id"],
    "stage": row["stage"],
    "task": row["task"],
    "example": row["example"],
    "runtime_seconds": f"{float(metrics.get('runtime_seconds', np.nan)):.3f}",
    "status": metrics.get("status", ""),
    "S": metrics.get("S", ""),
    "A": metrics.get("A", ""),
    "coverage_min": metrics.get("coverage_min", metrics.get("coverage_final", "")),
    "min_visits_min": metrics.get("min_visits_min", metrics.get("min_visits_final", "")),
    "final_residual_max": metrics.get("final_residual_max", metrics.get("ideal_final_residual", "")),
   }
  )
 return rows


def validation_report(root, manifest_rows, runs, failures, policy_rows, gap_rows, runtime, best_rows, grid_rows):
 optional_stages = {"stage3_seir_escalation"}
 done_ids = {run["row"]["run_id"] for run in runs}
 required_manifest_rows = [row for row in manifest_rows if row["stage"] not in optional_stages]
 optional_manifest_rows = [row for row in manifest_rows if row["stage"] in optional_stages]
 expected_ids = {row["run_id"] for row in required_manifest_rows}
 optional_ids = {row["run_id"] for row in optional_manifest_rows}
 failed_ids = {row.get("run_id", "") for row in failures}
 missing_rows = [
  {
   "run_id": row["run_id"],
   "stage": row["stage"],
   "task": row["task"],
   "example": row["example"],
   "manifest": row["manifest"],
  }
  for row in required_manifest_rows
  if row["run_id"] not in done_ids and row["run_id"] not in failed_ids
 ]
 optional_missing_rows = [
  {
   "run_id": row["run_id"],
   "stage": row["stage"],
   "task": row["task"],
   "example": row["example"],
   "manifest": row["manifest"],
  }
  for row in optional_manifest_rows
  if row["run_id"] not in done_ids and row["run_id"] not in failed_ids
 ]
 failed_rows = [
  {
   "run_id": row.get("run_id", ""),
   "stage": row.get("stage", ""),
   "task": row.get("task", ""),
   "example": row.get("example", ""),
   "error": row.get("error", ""),
   "run_dir": row.get("run_dir", ""),
  }
  for row in failures
 ]

 issues = []
 manifest_by_id = {row["run_id"]: row for row in manifest_rows}
 for run in runs:
  run_id = run["row"]["run_id"]
  manifest_row = manifest_by_id.get(run_id)
  if manifest_row is None:
   issues.append({"severity": "error", "check": "done_row_not_in_manifest", "run_id": run_id})
   continue
  mismatches = [
   field for field in MANIFEST_FIELDS
   if str(run["row"].get(field, "")) != str(manifest_row.get(field, ""))
  ]
  if mismatches:
   issues.append(
    {
     "severity": "error",
     "check": "done_row_manifest_mismatch",
     "run_id": run_id,
     "fields": mismatches[:12],
     "num_mismatched_fields": len(mismatches),
    }
   )

 for row in policy_rows:
  value = float(row["cost"])
  if not np.isfinite(value):
   issues.append({"severity": "error", "check": "finite_cost", "run_id": row["run_id"], "details": row})

 for row in runtime:
  stage = row["stage"]
  task = row["task"]
  if task == "idealized_profile" and row["final_residual_max"] != "":
   residual = float(row["final_residual_max"])
   if not np.isfinite(residual) or residual > 1e-9:
    severity = "warning" if stage == "stage0_smoke" else "error"
    issues.append(
     {
      "severity": severity,
      "check": "idealized_residual",
      "run_id": row["run_id"],
      "value": residual,
      "threshold": 1e-9,
     }
    )
  if task in {"sampled_profile", "convergence"}:
   if row["coverage_min"] != "":
    coverage = float(row["coverage_min"])
    if coverage < 100.0:
     severity = "warning" if stage == "stage0_smoke" else "error"
     issues.append(
      {
       "severity": severity,
       "check": "sampled_coverage",
       "run_id": row["run_id"],
       "value": coverage,
       "threshold": 100.0,
      }
     )
   if row["min_visits_min"] != "":
    min_visits = float(row["min_visits_min"])
    if min_visits <= 0.0:
     severity = "warning" if stage == "stage0_smoke" else "error"
     issues.append(
      {
       "severity": severity,
       "check": "sampled_min_visits",
       "run_id": row["run_id"],
       "value": min_visits,
       "threshold": "> 0",
      }
     )

 def gap_summary(example):
  rows = [
   row for row in gap_rows
   if row["stage"] == "stage3_final" and row["example"] == example
  ]
  if example == "seir":
   escalated = [
    row for row in gap_rows
    if row["stage"] == "stage3_seir_escalation" and row["example"] == example
   ]
   if escalated:
    rows = escalated
  if not rows:
   return None
  values = np.asarray([float(row["abs_gap"]) for row in rows], dtype=float)
  return float(np.mean(values)), float(np.max(values)), float(np.quantile(values, 0.95))

 for example in ("sis", "seir"):
  summary = gap_summary(example)
  if summary is None:
   continue
  mean_gap, max_gap, q95_gap = summary
  if mean_gap > 0.002:
   issues.append({"severity": "error", "check": f"{example}_mean_gap", "value": mean_gap, "threshold": 0.002})
  if q95_gap > 0.01:
   issues.append({"severity": "error", "check": f"{example}_q95_gap", "value": q95_gap, "threshold": 0.01})
  if max_gap > 0.01:
   issues.append({"severity": "warning", "check": f"{example}_max_gap_outlier", "value": max_gap, "threshold": 0.01})

 for example in ("sysrisk",):
  rows = [
   row for row in gap_rows
   if row["stage"] == "stage3_final"
   and row["example"] == example
   and float(row["M"]) >= 0.3
  ]
  if rows:
   values = np.asarray([float(row["abs_gap"]) for row in rows], dtype=float)
   if float(np.mean(values)) > 0.05:
    issues.append({"severity": "warning", "check": "sysrisk_robust_mean_gap", "value": float(np.mean(values)), "threshold": 0.05})

 final_sampled = [
  row for row in policy_rows
  if row["stage"] == "stage3_final" and row["solver"] == "sampled"
 ]
 if final_sampled:
  grouped = group_float_rows(final_sampled, ["example", "p", "M"], "cost")
  by_example_p = defaultdict(list)
  for (example, p_val, m_val), values in grouped.items():
   by_example_p[(example, p_val)].append((float(m_val), float(np.mean(values))))
  for example in ENV_ORDER:
   has_positive_improvement = False
   for (ex, _p_val), items in by_example_p.items():
    if ex != example:
     continue
    baselines = [cost for m_val, cost in items if abs(m_val) < 1e-12]
    if not baselines:
     continue
    baseline = baselines[0]
    best_m, best_cost = min(items, key=lambda item: item[1])
    if best_m > 0.0 and best_cost < baseline:
     has_positive_improvement = True
     break
   if not has_positive_improvement:
    issues.append({"severity": "warning", "check": "moderate_robustness_effect", "example": example})

 manifest_status = []
 by_stage = defaultdict(lambda: {"expected": 0, "done": 0, "failed": 0, "missing": 0})
 row_by_id = {row["run_id"]: row for row in manifest_rows}
 for row in manifest_rows:
  stage = row["stage"]
  by_stage[stage]["expected"] += 1
  if row["run_id"] in done_ids:
   by_stage[stage]["done"] += 1
  elif row["run_id"] in failed_ids:
   by_stage[stage]["failed"] += 1
  else:
   by_stage[stage]["missing"] += 1
 for stage, vals in sorted(by_stage.items()):
  manifest_status.append({"stage": stage, "optional": stage in optional_stages, **vals})

 payload = {
  "total_manifest_rows": len(manifest_rows),
  "expected_runs": len(expected_ids),
  "optional_expected_runs": len(optional_ids),
  "completed_runs": len(done_ids),
  "failed_runs": len(failed_ids),
  "missing_runs": len(missing_rows),
  "optional_missing_runs": len(optional_missing_rows),
  "manifest_status": manifest_status,
  "issues": issues,
  "error_count": sum(1 for item in issues if item["severity"] == "error"),
  "warning_count": sum(1 for item in issues if item["severity"] == "warning"),
 }
 validation_dir = root / "aggregates"
 write_csv(validation_dir / "missing_runs.csv", ["run_id", "stage", "task", "example", "manifest"], missing_rows)
 write_csv(validation_dir / "optional_missing_runs.csv", ["run_id", "stage", "task", "example", "manifest"], optional_missing_rows)
 write_csv(validation_dir / "failed_runs.csv", ["run_id", "stage", "task", "example", "error", "run_dir"], failed_rows)
 write_csv(validation_dir / "manifest_status.csv", ["stage", "optional", "expected", "done", "failed", "missing"], manifest_status)
 write_json(validation_dir / "validation_report.json", payload)
 return payload


def color_for_p(p_val):
 return plt.cm.viridis(0.08 + 0.84 * float(p_val))


def grid_key(row):
 return (int(row["n_disc"]), int(row["a_disc"]), float(row["discount"]))


def plot_main_profiles(policy_rows, gap_rows, out_dir):
 final = [r for r in policy_rows if r["stage"] == "stage3_final" and r["solver"] == "sampled"]
 if not final:
  final = [r for r in policy_rows if r["solver"] == "sampled"]
 ideals = [r for r in policy_rows if r["stage"] == "stage1_idealized" and r["solver"] == "idealized"]
 if not ideals:
  ideals = [r for r in policy_rows if r["solver"] == "idealized"]
 for example in ENV_ORDER:
  sampled = [r for r in final if r["example"] == example]
  if not sampled:
   continue
  sampled_grid = sorted({grid_key(r) for r in sampled})[0]
  fig, ax = plt.subplots(figsize=(8.8, 5.8))
  p_values = sorted({float(r["p"]) for r in sampled})
  for p_val in p_values:
   ms = sorted({float(r["M"]) for r in sampled if abs(float(r["p"]) - p_val) < 1e-12})
   means, lows, highs = [], [], []
   for m_val in ms:
    vals = [
     -float(r["cost"])
     for r in sampled
     if abs(float(r["p"]) - p_val) < 1e-12 and abs(float(r["M"]) - m_val) < 1e-12
    ]
    mean, lo, hi = bootstrap_ci(vals)
    means.append(mean)
    lows.append(lo)
    highs.append(hi)
   color = color_for_p(p_val)
   ax.plot(ms, means, color=color, marker="o", linewidth=2.0, label=rf"$\zeta={p_val:g}$")
   ax.fill_between(ms, lows, highs, color=color, alpha=0.13, linewidth=0)

   ideal_candidates = [r for r in ideals if r["example"] == example and abs(float(r["p"]) - p_val) < 1e-12]
   same_grid_ideal = [r for r in ideal_candidates if grid_key(r) == sampled_grid]
   if same_grid_ideal:
    ideal_candidates = same_grid_ideal
   if ideal_candidates:
    ideal_by_m = defaultdict(list)
    for r in ideal_candidates:
     ideal_by_m[float(r["M"])].append(-float(r["cost"]))
    ideal_m = sorted(ideal_by_m)
    ideal_reward = [np.mean(ideal_by_m[m]) for m in ideal_m]
    ax.plot(ideal_m, ideal_reward, color=color, linestyle="--", linewidth=1.5, alpha=0.75)

  ax.set_xlabel("robustness radius m")
  ax.set_ylabel("expected discounted reward")
  ax.grid(True, linestyle="--", alpha=0.3)
  ax.legend(title=r"eval. drift $\zeta$", bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=True)
  fig.tight_layout()
  for ext in ("png", "pdf"):
   fig.savefig(out_dir / f"main_profiles_{example}.{ext}", dpi=220, bbox_inches="tight")
  plt.close(fig)


def plot_relative_improvements(rows, out_dir):
 for example in ENV_ORDER:
  subset = [r for r in rows if r["example"] == example]
  if not subset:
   continue
  fig, ax = plt.subplots(figsize=(8.4, 5.4))
  for p_val in sorted({float(r["p"]) for r in subset}):
   rows_p = [r for r in subset if abs(float(r["p"]) - p_val) < 1e-12]
   rows_p.sort(key=lambda r: float(r["M"]))
   ax.plot(
    [float(r["M"]) for r in rows_p],
    [100.0 * float(r["relative_improvement_mean"]) for r in rows_p],
    marker="o",
    color=color_for_p(p_val),
    label=f"p={p_val:g}",
   )
  ax.axhline(0.0, color="black", linewidth=0.8)
  ax.set_xlabel("robustness radius m")
  ax.set_ylabel("relative improvement (%)")
  ax.grid(True, linestyle="--", alpha=0.3)
  ax.legend(title=r"eval. drift $\zeta$", bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=True)
  fig.tight_layout()
  for ext in ("png", "pdf"):
   fig.savefig(out_dir / f"relative_improvement_{example}.{ext}", dpi=220, bbox_inches="tight")
  plt.close(fig)


def plot_idealized_landscapes(policy_rows, out_dir):
 for example in ENV_ORDER:
  subset = [
   r
   for r in policy_rows
   if r["example"] == example and r["stage"] == "stage1_idealized" and r["solver"] == "idealized"
  ]
  if not subset:
   subset = [
    r
    for r in policy_rows
    if r["example"] == example and r["solver"] == "idealized"
   ]
  if not subset:
   continue
  available_grids = sorted({grid_key(r) for r in subset})
  preferred_grid = (5, 5, 0.5) if example == "sysrisk" and (5, 5, 0.5) in available_grids else available_grids[0]
  subset = [r for r in subset if grid_key(r) == preferred_grid]
  p_values = sorted({float(r["p"]) for r in subset})
  m_values = sorted({float(r["M"]) for r in subset})
  matrix = np.full((len(p_values), len(m_values)), np.nan)
  for r in subset:
   matrix[p_values.index(float(r["p"])), m_values.index(float(r["M"]))] = -float(r["cost"])
  fig, ax = plt.subplots(figsize=(8.0, 5.2))
  im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="plasma", extent=[min(m_values), max(m_values), min(p_values), max(p_values)])
  ax.set_xlabel("robustness radius m")
  ax.set_ylabel(r"eval. drift $\zeta$")
  fig.colorbar(im, ax=ax, label="expected discounted reward")
  fig.tight_layout()
  for ext in ("png", "pdf"):
   fig.savefig(out_dir / f"idealized_landscape_{example}.{ext}", dpi=220, bbox_inches="tight")
  plt.close(fig)


def plot_hyperparameter(rows, out_dir):
 if not rows:
  return
 fig, ax = plt.subplots(figsize=(7.2, 4.8))
 by_w = defaultdict(list)
 for r in rows:
  if abs(float(r["M"])) > 1e-12:
   continue
  by_w[float(r["w_lr"])].append((int(float(r["num_updates"])), float(r["median_abs_gap_p1"])))
 for w_lr, vals in sorted(by_w.items()):
  vals.sort()
  ax.plot([v[0] for v in vals], [v[1] for v in vals], marker="o", linewidth=2.0, label=f"w={w_lr:g}")
 ax.set_xscale("log")
 ax.set_xlabel("updates per m")
 ax.set_ylabel("median sampled-vs-idealized abs gap")
 ax.grid(True, which="both", linestyle="--", alpha=0.3)
 ax.legend(frameon=True)
 fig.tight_layout()
 for ext in ("png", "pdf"):
  fig.savefig(out_dir / f"hyperparameter_screen_sysrisk.{ext}", dpi=220, bbox_inches="tight")
 plt.close(fig)


def plot_policy_disagreement(rows, out_dir):
 final = [r for r in rows if r["stage"] == "stage3_final"]
 if not final:
  return
 grouped = defaultdict(list)
 for r in final:
  grouped[(r["example"], r["M"])].append(float(r["occupancy_disagreement"]))
 fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), sharey=True)
 for ax, example in zip(axes, ENV_ORDER):
  items = sorted((float(m), vals) for (ex, m), vals in grouped.items() if ex == example)
  if not items:
   continue
  ax.bar([x[0] for x in items], [np.mean(x[1]) for x in items], width=0.035 if example != "sysrisk" else 0.06)
  ax.set_title(example_label(example))
  ax.set_xlabel("M")
  ax.grid(True, axis="y", linestyle="--", alpha=0.3)
 axes[0].set_ylabel("occupancy-weighted disagreement")
 fig.tight_layout()
 for ext in ("png", "pdf"):
  fig.savefig(out_dir / f"policy_disagreement.{ext}", dpi=220, bbox_inches="tight")
 plt.close(fig)


def aggregate_curve(rows, value_field):
 grouped = defaultdict(list)
 for r in rows:
  value = r.get(value_field, "")
  if value == "":
   continue
  grouped[float(r["step"])].append(float(value))
 steps = np.array(sorted(grouped), dtype=float)
 if steps.size == 0:
  return steps, steps, steps, steps
 med = np.array([np.median(grouped[x]) for x in steps], dtype=float)
 lo = np.array([np.quantile(grouped[x], 0.1) for x in steps], dtype=float)
 hi = np.array([np.quantile(grouped[x], 0.9) for x in steps], dtype=float)
 return steps, med, lo, hi


def aggregate_relative_policy_gap_pct(rows):
 grouped = defaultdict(list)
 for r in rows:
  gap = r.get("policy_gap_abs_p1", "")
  denom = r.get("reference_cost_p1", "")
  if gap == "" or denom == "":
   continue
  denom = float(denom)
  if not np.isfinite(denom) or denom <= 0.0:
   continue
  grouped[float(r["step"])].append(100.0 * float(gap) / denom)
 steps = np.array(sorted(grouped), dtype=float)
 if steps.size == 0:
  return steps, steps, steps, steps
 med = np.array([np.median(grouped[x]) for x in steps], dtype=float)
 lo = np.array([np.quantile(grouped[x], 0.1) for x in steps], dtype=float)
 hi = np.array([np.quantile(grouped[x], 0.9) for x in steps], dtype=float)
 return steps, med, lo, hi


def convergence_color(m_val):
 return COLORS.get(round(float(m_val), 3), plt.cm.tab10(abs(hash(round(float(m_val), 3))) % 10))


def positive_for_log(values):
 arr = np.asarray(values, dtype=float).copy()
 arr[~np.isfinite(arr)] = np.nan
 arr[arr <= 0.0] = np.nan
 return arr


def plot_nontruncated_gap_figure(preferred, out_dir, filename, ylabel, title, curve_getter):
 fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=False)
 for ax, example in zip(axes, ENV_ORDER):
  rows_ex = [r for r in preferred if r["example"] == example]
  if not rows_ex:
   ax.set_axis_off()
   continue
  median_values = []
  upper_values = []
  for m_val in sorted({float(r["M"]) for r in rows_ex}):
   rows_m = [r for r in rows_ex if abs(float(r["M"]) - m_val) < 1e-12]
   steps, med, lo, hi = curve_getter(rows_m)
   if steps.size == 0:
    continue
   med_plot = positive_for_log(med)
   lo_plot = positive_for_log(lo)
   hi_plot = positive_for_log(hi)
   median_values.extend(med_plot[np.isfinite(med_plot)].tolist())
   upper_values.extend(hi_plot[np.isfinite(hi_plot)].tolist())
   color = convergence_color(m_val)
   ax.plot(steps, med_plot, color=color, linewidth=2.0, label=f"M={m_val:g}")
   ax.fill_between(steps, lo_plot, hi_plot, color=color, alpha=0.15, linewidth=0)
  ax.set_title(example_label(example))
  ax.set_xlabel("sampled updates")
  ax.set_xscale("log")
  ax.set_yscale("log")
  if median_values:
   upper = max(upper_values) if upper_values else max(median_values)
   ax.set_ylim(min(median_values) * 0.5, upper * 1.35)
  ax.grid(True, which="both", linestyle="--", alpha=0.3)
  ax.legend(frameon=True, fontsize=8)
 axes[0].set_ylabel(ylabel)
 fig.tight_layout()
 for ext in ("png", "pdf"):
  fig.savefig(out_dir / f"{filename}.{ext}", dpi=220, bbox_inches="tight")
 plt.close(fig)


def plot_convergence(conv_rows, out_dir):
 if not conv_rows:
  return
 preferred = [r for r in conv_rows if r["study"] == "main"]
 if not preferred:
  preferred = conv_rows
 for value_field, ylabel, filename in [
  ("q_sup_error", r"$\|\check Q_T-\check Q^*_m\|_\infty$", "convergence_q_error"),
 ]:
  fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=False)
  for ax, example in zip(axes, ENV_ORDER):
   rows_ex = [r for r in preferred if r["example"] == example]
   if not rows_ex:
    ax.set_axis_off()
    continue
   guide_drawn = False
   for m_val in sorted({float(r["M"]) for r in rows_ex}):
    rows_m = [r for r in rows_ex if abs(float(r["M"]) - m_val) < 1e-12]
    steps, med, lo, hi = aggregate_curve(rows_m, value_field)
    if steps.size == 0:
     continue
    med = np.maximum(med, 1e-12)
    lo = np.maximum(lo, 1e-12)
    hi = np.maximum(hi, 1e-12)
    color = convergence_color(m_val)
    ax.plot(steps, med, color=color, linewidth=2.0, label=rf"$m={m_val:g}$")
    ax.fill_between(steps, lo, hi, color=color, alpha=0.15, linewidth=0)
    if not guide_drawn and steps.size > 1:
     w_lr = float(rows_m[0].get("w_lr", 0.7) or 0.7)
     guide = np.sqrt(np.log(np.maximum(steps, 3.0)) / np.maximum(steps, 1.0) ** w_lr)
     guide = guide / guide[0] * med[0]
     ax.plot(
      steps,
      guide,
      color="black",
      linestyle="--",
      linewidth=1.3,
      alpha=0.65,
      label=rf"$\sqrt{{\log T/T^{{{w_lr:.2g}}}}}$",
     )
     guide_drawn = True
   ax.set_title(example_label(example))
   ax.set_xlabel("sampled updates")
   ax.set_xscale("log")
   ax.set_yscale("log")
   ax.grid(True, which="both", linestyle="--", alpha=0.3)
   ax.legend(frameon=True, fontsize=8)
  axes[0].set_ylabel(ylabel)
  fig.tight_layout()
  for ext in ("png", "pdf"):
   fig.savefig(out_dir / f"{filename}.{ext}", dpi=220, bbox_inches="tight")
  plt.close(fig)

 plot_nontruncated_gap_figure(
  preferred,
  out_dir,
  "convergence_policy_gap",
  "absolute policy-cost gap at p=1",
  "Greedy-policy performance gap",
  lambda rows_m: aggregate_curve(rows_m, "policy_gap_abs_p1"),
 )
 plot_nontruncated_gap_figure(
  preferred,
  out_dir,
  "convergence_policy_gap_relative",
  "relative cost gap at p=1 (%)",
  "Relative greedy-policy performance gap",
  aggregate_relative_policy_gap_pct,
 )

 fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=False)
 for ax, example in zip(axes, ENV_ORDER):
  rows_ex = [r for r in preferred if r["example"] == example]
  if not rows_ex:
   ax.set_axis_off()
   continue
  steps, med, lo, hi = aggregate_curve(rows_ex, "min_visits")
  ax.plot(steps, med, color="#1f78b4", linewidth=2.0, label="min visits")
  ax.fill_between(steps, lo, hi, color="#1f78b4", alpha=0.12, linewidth=0)
  ax2 = ax.twinx()
  _, cov_med, cov_lo, cov_hi = aggregate_curve(rows_ex, "coverage_pct")
  ax2.plot(steps, cov_med, color="#66a61e", linestyle="--", linewidth=1.8, label="coverage")
  ax2.fill_between(steps, cov_lo, cov_hi, color="#66a61e", alpha=0.10, linewidth=0)
  ax.set_title(example_label(example))
  ax.set_xlabel("sampled updates")
  ax.set_xscale("log")
  ax.set_ylabel("minimum visits")
  ax2.set_ylabel("coverage (%)")
  ax2.set_ylim(0, 105)
  ax.grid(True, which="both", linestyle="--", alpha=0.3)
 fig.tight_layout()
 for ext in ("png", "pdf"):
  fig.savefig(out_dir / f"convergence_coverage.{ext}", dpi=220, bbox_inches="tight")
 plt.close(fig)


def plot_theorem_rate(conv_rows, out_dir):
 theorem = [r for r in conv_rows if r["study"] == "theorem"]
 if not theorem:
  return
 fig, ax = plt.subplots(figsize=(8.2, 5.0))
 colors = {"sysrisk": "#1b9e77", "sis": "#d95f02", "seir": "#7570b3"}
 for example in ENV_ORDER:
  rows_ex = [r for r in theorem if r["example"] == example]
  if not rows_ex:
   continue
  by_seed = defaultdict(list)
  for r in rows_ex:
   by_seed[r["seed"]].append(r)
  normalized = []
  steps = None
  for seed_rows in by_seed.values():
   seed_rows.sort(key=lambda r: float(r["step"]))
   vals = np.maximum(np.array([float(r["q_sup_error"]) for r in seed_rows], dtype=float), 1e-12)
   normalized.append(vals / vals[0])
   if steps is None:
    steps = np.array([float(r["step"]) for r in seed_rows], dtype=float)
  arr = np.asarray(normalized, dtype=float)
  med = np.median(arr, axis=0)
  lo = np.quantile(arr, 0.1, axis=0)
  hi = np.quantile(arr, 0.9, axis=0)
  ax.plot(steps, med, color=colors[example], linewidth=2.2, label=example_label(example))
  ax.fill_between(steps, lo, hi, color=colors[example], alpha=0.15, linewidth=0)
  guide = np.sqrt(np.log(np.maximum(steps, 3.0)) / np.maximum(steps, 1.0) ** 0.7)
  guide = guide / guide[0]
  ax.plot(steps, guide, color="black", linestyle="--", linewidth=1.3, alpha=0.65)
 ax.set_xscale("log")
 ax.set_yscale("log")
 ax.set_xlabel("sampled updates")
 ax.set_ylabel(r"normalized $\|\check Q_T-\check Q^*\|_\infty$")
 ax.grid(True, which="both", linestyle="--", alpha=0.3)
 ax.legend(frameon=True)
 fig.tight_layout()
 for ext in ("png", "pdf"):
  fig.savefig(out_dir / f"convergence_theorem_rate.{ext}", dpi=220, bbox_inches="tight")
 plt.close(fig)


def plot_grid_sensitivity(rows, out_dir):
 if not rows:
  return
 fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), sharey=False)
 for ax, example in zip(axes, ENV_ORDER):
  rows_ex = [r for r in rows if r["example"] == example and abs(float(r["p"]) - 0.5) < 1e-12]
  if not rows_ex:
   ax.set_axis_off()
   continue
  labels = [f"n={r['n_disc']}, a={r['a_disc']}" for r in rows_ex]
  ax.bar(np.arange(len(rows_ex)), [float(r["best_M"]) for r in rows_ex], color="#4c78a8")
  ax.set_title(example_label(example))
  ax.set_ylabel("best M at p=0.5")
  ax.set_xticks(np.arange(len(rows_ex)))
  ax.set_xticklabels(labels, rotation=35, ha="right")
  ax.grid(True, axis="y", linestyle="--", alpha=0.3)
 fig.tight_layout()
 for ext in ("png", "pdf"):
  fig.savefig(out_dir / f"grid_sensitivity.{ext}", dpi=220, bbox_inches="tight")
 plt.close(fig)


def write_report_tables(best_rows, hyper_rows, runtime_rows, out_dir):
 out_dir.mkdir(parents=True, exist_ok=True)
 with (out_dir / "best_radius_by_p.tex").open("w") as handle:
  handle.write("\\begin{tabular}{llrrr}\n\\toprule\n")
  handle.write("Example & $p$ & Best $M$ & Best cost & Rel. improvement \\\\\n\\midrule\n")
  for r in best_rows:
   handle.write(
    f"{example_label(r['example'])} & {float(r['p']):.3g} & {float(r['best_M']):.3g} & "
    f"{float(r['best_cost_mean']):.4g} & {100.0 * float(r['relative_improvement_vs_M0']):.2f}\\% \\\\\n"
   )
  handle.write("\\bottomrule\n\\end{tabular}\n")

 with (out_dir / "runtime_summary.tex").open("w") as handle:
  handle.write("\\begin{tabular}{llrr}\n\\toprule\n")
  handle.write("Stage & Task & Runs & Total hours \\\\\n\\midrule\n")
  grouped = defaultdict(list)
  for r in runtime_rows:
   grouped[(r["stage"], r["task"])].append(float(r["runtime_seconds"]))
  for (stage, task), vals in sorted(grouped.items()):
   handle.write(f"{stage} & {task} & {len(vals)} & {sum(vals)/3600.0:.2f} \\\\\n")
  handle.write("\\bottomrule\n\\end{tabular}\n")


def write_hyperparameter_recommendation(hyper_rows, out_dir):
 if not hyper_rows:
  return {}
 by_candidate = defaultdict(dict)
 for row in hyper_rows:
  key = (float(row["w_lr"]), int(float(row["num_updates"])))
  by_candidate[key][float(row["M"])] = float(row["median_abs_gap_p1"])
 candidates = []
 for key, by_m in by_candidate.items():
  if 0.0 not in by_m:
   continue
  candidates.append(
   {
    "w_lr": key[0],
    "num_updates": key[1],
    "median_abs_gap_m0_p1": by_m.get(0.0, math.inf),
    "median_abs_gap_m06_p1": by_m.get(0.6, math.inf),
   }
  )
 if not candidates:
  return {}
 best = min(candidates, key=lambda x: (x["median_abs_gap_m0_p1"], x["median_abs_gap_m06_p1"]))
 default = next(
  (x for x in candidates if abs(x["w_lr"] - 0.8) < 1e-12 and x["num_updates"] == 10_000_000),
  None,
 )
 selected = best
 reason = "best median adverse-drift gap at M=0, tie-broken by M=0.6"
 if default is not None:
  default_close = (
   default["median_abs_gap_m0_p1"] <= 1.05 * best["median_abs_gap_m0_p1"]
   and default["median_abs_gap_m06_p1"] <= 1.05 * best["median_abs_gap_m06_p1"]
  )
  if default_close:
   selected = default
   reason = "default w=0.8, 10M is within 5 percent of the best screen score"
 payload = {
  "selected": selected,
  "best": best,
  "reason": reason,
  "candidates": sorted(candidates, key=lambda x: (x["w_lr"], x["num_updates"])),
 }
 write_json(out_dir / "systemic_hyperparameter_recommendation.json", payload)
 return payload


def main():
 parser = argparse.ArgumentParser()
 parser.add_argument("--campaign-root", type=str, default=str(CAMPAIGN_DIR))
 args = parser.parse_args()
 root = Path(args.campaign_root).resolve()
 agg = root / "aggregates"
 figs = root / "figures"
 tables = root / "report_tables"
 agg.mkdir(parents=True, exist_ok=True)
 figs.mkdir(parents=True, exist_ok=True)
 tables.mkdir(parents=True, exist_ok=True)

 manifest_rows = read_manifest_rows(root)
 runs = discover_runs(root)
 failures = discover_failures(root)
 refs = reference_runs(runs)
 policy = profile_rows(runs)
 gaps = sampled_idealized_gaps(runs, refs)
 best = best_radius_rows(policy)
 rel = relative_improvement_rows(policy)
 hyper = hyperparameter_rows(gaps)
 gap_summary = gap_summary_rows(gaps)
 gap_outliers = gap_outlier_rows(gaps)
 conv = convergence_rows(runs)
 diag = policy_diagnostics_rows(runs, refs)
 grid = grid_sensitivity_rows(policy)
 runtime = runtime_rows(runs)

 write_csv(agg / "final_policy_costs.csv", [
  "run_id", "stage", "task", "solver", "example", "n_disc", "a_disc", "discount", "seed", "w_lr", "num_updates", "p", "M", "cost"
 ], policy)
 write_csv(agg / "sampled_vs_idealized_gaps.csv", [
  "run_id", "stage", "example", "n_disc", "a_disc", "discount", "seed", "w_lr", "num_updates", "p", "M", "sampled_cost", "idealized_cost", "gap", "abs_gap", "rel_gap", "reference_run_id"
 ], gaps)
 write_csv(agg / "best_radius_by_p.csv", [
  "example", "p", "best_M", "best_cost_mean", "best_cost_ci_low", "best_cost_ci_high", "M0_cost_mean", "relative_improvement_vs_M0", "n_seeds"
 ], best)
 write_csv(agg / "relative_improvements.csv", [
  "example", "p", "M", "relative_improvement_mean", "relative_improvement_ci_low", "relative_improvement_ci_high", "n_seeds"
 ], rel)
 write_csv(agg / "hyperparameter_screen.csv", [
  "example", "w_lr", "num_updates", "M", "median_abs_gap_p1", "mean_abs_gap_p1", "n_seeds"
 ], hyper)
 write_csv(agg / "gap_summary_by_stage_example.csv", [
  "stage", "example", "n", "mean_abs_gap", "max_abs_gap", "q90_abs_gap", "q95_abs_gap", "q99_abs_gap"
 ], gap_summary)
 write_csv(agg / "gap_outliers.csv", [
  "run_id", "stage", "example", "n_disc", "a_disc", "discount", "seed", "w_lr", "num_updates",
  "p", "M", "sampled_cost", "idealized_cost", "gap", "abs_gap", "rel_gap", "reference_run_id",
  "outlier_threshold"
 ], gap_outliers)
 write_csv(agg / "convergence_summary.csv", [
  "run_id", "stage", "study", "example", "n_disc", "a_disc", "discount", "M", "seed", "w_lr", "num_updates", "step", "q_sup_error", "q_mae_error", "bellman_residual_inf", "coverage_pct", "min_visits", "mean_visits", "policy_gap_abs_p0", "policy_gap_abs_p05", "policy_gap_abs_p1", "policy_cost_p1", "reference_cost_p1"
 ], conv)
 write_csv(agg / "policy_margin_disagreement.csv", [
  "run_id", "stage", "example", "n_disc", "a_disc", "discount", "seed", "M", "reference_run_id", "uniform_disagreement", "occupancy_disagreement", "margin_min", "margin_median", "margin_mean", "margin_occ_mean", "margin_share_lt_1e_4", "margin_share_lt_1e_3", "margin_share_lt_1e_2"
 ], diag)
 write_csv(agg / "grid_sensitivity.csv", ["example", "n_disc", "a_disc", "p", "best_M", "best_cost"], grid)
 write_csv(agg / "runtime_summary.csv", [
  "run_id", "stage", "task", "example", "runtime_seconds", "status", "S", "A", "coverage_min", "min_visits_min", "final_residual_max"
 ], runtime)

 plot_main_profiles(policy, gaps, figs)
 plot_relative_improvements(rel, figs)
 plot_idealized_landscapes(policy, figs)
 plot_hyperparameter(hyper, figs)
 plot_policy_disagreement(diag, figs)
 plot_convergence(conv, figs)
 plot_theorem_rate(conv, figs)
 plot_grid_sensitivity(grid, figs)
 write_report_tables(best, hyper, runtime, tables)
 recommendation = write_hyperparameter_recommendation(hyper, agg)
 validation = validation_report(root, manifest_rows, runs, failures, policy, gaps, runtime, best, grid)

 status = {
  "total_manifest_rows": validation["total_manifest_rows"],
  "expected_runs": validation["expected_runs"],
  "optional_expected_runs": validation["optional_expected_runs"],
  "completed_runs": len(runs),
  "failed_runs": validation["failed_runs"],
  "missing_runs": validation["missing_runs"],
  "optional_missing_runs": validation["optional_missing_runs"],
  "policy_rows": len(policy),
  "gap_rows": len(gaps),
  "best_radius_rows": len(best),
  "relative_improvement_rows": len(rel),
  "hyperparameter_rows": len(hyper),
  "gap_summary_rows": len(gap_summary),
  "gap_outlier_rows": len(gap_outliers),
  "convergence_rows": len(conv),
  "policy_diagnostic_rows": len(diag),
  "grid_sensitivity_rows": len(grid),
  "has_hyperparameter_recommendation": bool(recommendation),
  "validation_error_count": validation["error_count"],
  "validation_warning_count": validation["warning_count"],
 }
 write_json(agg / "collection_summary.json", status)
 print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
 main()
