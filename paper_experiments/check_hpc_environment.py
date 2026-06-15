#!/usr/bin/env python3
import csv
import importlib
import json
import os
import platform
import sys
from pathlib import Path

CAMPAIGN_DIR = Path(__file__).resolve().parent
ROOT = CAMPAIGN_DIR.parents[1]
if str(ROOT) not in sys.path:
 sys.path.insert(0, str(ROOT))

_CACHE_DIR = CAMPAIGN_DIR / ".cache"
_CACHE_DIR.mkdir(exist_ok=True)
(_CACHE_DIR / "matplotlib").mkdir(exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def module_version(name):
 module = importlib.import_module(name)
 return str(getattr(module, "__version__", "unknown"))


def read_manifests():
 from campaign_lib import MANIFEST_FIELDS

 rows = []
 issues = []
 for path in sorted((CAMPAIGN_DIR / "manifests").glob("stage*.csv")):
  with path.open(newline="") as handle:
   reader = csv.DictReader(handle)
   if reader.fieldnames != MANIFEST_FIELDS:
    issues.append(
     {
      "severity": "error",
      "check": "manifest_columns",
      "manifest": path.name,
      "fieldnames": reader.fieldnames,
     }
    )
   for idx, row in enumerate(reader):
    item = dict(row)
    item["manifest"] = path.name
    item["manifest_row_index"] = idx
    rows.append(item)
 return rows, issues


def validate_manifest_rows(rows):
 issues = []
 seen = {}
 for row in rows:
  run_id = row.get("run_id", "")
  if not run_id:
   issues.append({"severity": "error", "check": "empty_run_id", "manifest": row.get("manifest", "")})
   continue
  if run_id in seen:
   issues.append(
    {
     "severity": "error",
     "check": "duplicate_run_id",
     "run_id": run_id,
     "first_manifest": seen[run_id],
     "second_manifest": row.get("manifest", ""),
    }
   )
  seen[run_id] = row.get("manifest", "")

  if row.get("stage") != "stage0_smoke" and row.get("task") in {"sampled_profile", "convergence"}:
   if str(row.get("coverage_passes", "0")) not in {"", "0", "0.0"}:
    issues.append(
     {
      "severity": "error",
      "check": "coverage_warm_start_enabled",
      "run_id": run_id,
      "coverage_passes": row.get("coverage_passes", ""),
     }
    )
 return issues


def smoke_import_and_target():
 import numpy as np

 from campaign_lib import P_HAT, lambda_grid_from_name, make_tables_for_row
 from engine.robust_solvers import compute_robust_bellman_target

 row = {
  "example": "sis",
  "n_disc": "4",
  "a_disc": "3",
  "cost_dist": "0.5",
  "c_f": "1.8",
 }
 _, tables = make_tables_for_row(row)
 q_values = np.zeros((tables["S"], tables["A"]), dtype=float)
 target = compute_robust_bellman_target(
  q_values,
  tables,
  robust_m=0.05,
  q_norm=1,
  p_hat=P_HAT,
  lambda_grid=lambda_grid_from_name("default"),
  discount=0.5,
 )
 if target.shape != (tables["S"], tables["A"]):
  raise RuntimeError(f"unexpected target shape {target.shape}")
 if not np.all(np.isfinite(target)):
  raise RuntimeError("non-finite smoke Bellman target")
 return {
  "example": "sis",
  "S": int(tables["S"]),
  "A": int(tables["A"]),
  "target_min": float(np.min(target)),
  "target_max": float(np.max(target)),
 }


def main():
 versions = {name: module_version(name) for name in ("numpy", "scipy", "matplotlib")}
 from campaign_lib import write_json

 for directory in ("runs", "logs", "aggregates", "figures", "report_tables"):
  (CAMPAIGN_DIR / directory).mkdir(parents=True, exist_ok=True)

 manifest_rows, issues = read_manifests()
 issues.extend(validate_manifest_rows(manifest_rows))
 smoke = smoke_import_and_target()
 payload = {
  "status": "ok" if not any(item["severity"] == "error" for item in issues) else "failed",
  "python": sys.version,
  "python_executable": sys.executable,
  "platform": platform.platform(),
  "package_versions": versions,
  "thread_env": {
   "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", ""),
   "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
   "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
  },
  "campaign_dir": str(CAMPAIGN_DIR),
  "manifest_rows": len(manifest_rows),
  "smoke_target": smoke,
  "issues": issues,
 }
 write_json(CAMPAIGN_DIR / "aggregates" / "environment_check.json", payload)
 print(json.dumps(payload, indent=2, sort_keys=True))
 if payload["status"] != "ok":
  raise SystemExit(1)


if __name__ == "__main__":
 main()
