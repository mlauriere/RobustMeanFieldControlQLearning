import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

CAMPAIGN_DIR = Path(__file__).resolve().parent
ROOT = CAMPAIGN_DIR.parents[1]
if str(ROOT) not in sys.path:
 sys.path.insert(0, str(ROOT))

from engine.convergence import ( # noqa: E402
 compute_policy_costs,
 make_log_checkpoints,
 run_sampled_qlearning_trace,
 run_value_iteration_trace,
)
from engine.evaluation import evaluate_policy_exact # noqa: E402
from engine.robust_solvers import ( # noqa: E402
 compute_robust_bellman_target,
 compute_single_robust_target,
 make_lambda_grid,
)
from engine.table_builder import precompute_tables # noqa: E402
from examples.seir.seir_env import SEIRModularEnv # noqa: E402
from examples.sis.sis_env import SISModularEnv # noqa: E402
from examples.systemic_risk.sysrisk_env import SysRiskModularEnv # noqa: E402


P_HAT = np.array([0.1, 0.8, 0.1], dtype=float)
BETAS = np.array([0.5, 0.81, 1.8], dtype=float)
SYS_RISK_NOISE = np.array([-1.0, 0.0, 1.0], dtype=float)
P_FINE = np.array([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0])
P_COARSE = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
M_SYS = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0])
M_EPI = np.array(
 [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]
)

MANIFEST_FIELDS = [
 "run_id",
 "stage",
 "task",
 "example",
 "n_disc",
 "a_disc",
 "discount",
 "q_norm",
 "m_grid",
 "p_grid",
 "num_updates",
 "seed",
 "w_lr",
 "coverage_passes",
 "interleaved_coverage_passes",
 "interleaved_coverage_interval",
 "n_checkpoints",
 "idealized_max_iter",
 "idealized_tol",
 "lambda_grid",
 "save_q",
 "cliff",
 "cost_dist",
 "dist_eff",
 "c_f",
 "notes",
]


def ensure_single_thread_env():
 os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
 os.environ.setdefault("OMP_NUM_THREADS", "1")
 os.environ.setdefault("MKL_NUM_THREADS", "1")


def timestamp():
 return time.strftime("%Y%m%d-%H%M%S")


def grid_to_str(values):
 return ",".join(f"{float(x):.12g}" for x in np.asarray(values, dtype=float))


def parse_float_list(value, default=None):
 if value is None or str(value).strip() == "":
  return np.asarray([] if default is None else default, dtype=float)
 return np.asarray([float(x.strip()) for x in str(value).split(",") if x.strip()], dtype=float)


def row_float(row, key, default):
 value = row.get(key, "")
 if value is None or str(value).strip() == "":
  return float(default)
 return float(value)


def row_int(row, key, default):
 value = row.get(key, "")
 if value is None or str(value).strip() == "":
  return int(default)
 return int(float(value))


def row_bool(row, key, default=False):
 value = str(row.get(key, "")).strip().lower()
 if value == "":
  return bool(default)
 return value in {"1", "true", "yes", "y"}


def read_manifest_row(manifest_path, row_index):
 with Path(manifest_path).open(newline="") as handle:
  rows = list(csv.DictReader(handle))
 idx = int(row_index)
 if idx < 0 or idx >= len(rows):
  raise IndexError(f"row_index {idx} is outside manifest with {len(rows)} rows")
 return rows[idx], len(rows)


def write_json(path, payload):
 path = Path(path)
 path.parent.mkdir(parents=True, exist_ok=True)
 with path.open("w") as handle:
  json.dump(payload, handle, indent=2, sort_keys=True)


def write_csv(path, fieldnames, rows):
 path = Path(path)
 path.parent.mkdir(parents=True, exist_ok=True)
 with path.open("w", newline="") as handle:
  writer = csv.DictWriter(handle, fieldnames=fieldnames)
  writer.writeheader()
  for row in rows:
   writer.writerow({field: row.get(field, "") for field in fieldnames})


def example_label(example):
 return {
  "sysrisk": "Systemic Risk",
  "sis": "SIS",
  "seir": "SEIR",
 }[example]


def default_noise_grid(example):
 return SYS_RISK_NOISE.copy() if example == "sysrisk" else BETAS.copy()


def adverse_law(example):
 return np.array([1.0, 0.0, 0.0]) if example == "sysrisk" else np.array([0.0, 0.0, 1.0])


def make_true_law(example, p_value):
 return (1.0 - float(p_value)) * P_HAT + float(p_value) * adverse_law(example)


def make_eval_laws(example):
 adv = adverse_law(example)
 return {
  "p0": P_HAT.copy(),
  "p05": 0.5 * P_HAT + 0.5 * adv,
  "p1": adv.copy(),
 }


def make_env(example, row):
 if example == "sysrisk":
  return SysRiskModularEnv(cliff_penalty=row_float(row, "cliff", 2.0))
 if example == "sis":
  return SISModularEnv(
   cost_distancing=row_float(row, "cost_dist", 0.5),
   lipschitz_constant=row_float(row, "c_f", 1.8),
  )
 if example == "seir":
  return SEIRModularEnv(
   cost_distancing=row_float(row, "cost_dist", 0.85),
   distancing_efficacy=row_float(row, "dist_eff", 1.0),
   lipschitz_constant=row_float(row, "c_f", 3.2),
  )
 raise ValueError(f"unknown example {example!r}")


def make_tables_for_row(row):
 example = row["example"]
 env = make_env(example, row)
 return env, precompute_tables(
  env,
  n_disc=row_int(row, "n_disc", 0),
  a_disc=row_int(row, "a_disc", 0),
  noise_grid=default_noise_grid(example),
 )


def lambda_grid_from_name(name):
 name = str(name or "default").strip().lower()
 if name in {"", "default"}:
  return make_lambda_grid()
 if name == "dense":
  linear = np.linspace(0.0, 150.0, 76)
  log = np.geomspace(0.25, 3000.0, 60)
  return np.unique(np.concatenate(([0.0], linear, log)))
 if name.startswith("max:"):
  return make_lambda_grid(float(name.split(":", 1)[1]))
 values = parse_float_list(name)
 if values.size == 0:
  raise ValueError(f"invalid lambda grid {name!r}")
 return np.unique(values)


def run_async_qlearning_with_stats(
 tables,
 robust_m,
 discount,
 q_norm,
 p_hat,
 lambda_grid,
 num_updates,
 seed,
 w_lr,
 coverage_passes=0,
 interleaved_coverage_passes=0,
 interleaved_coverage_interval=0,
):
 S, A = tables["S"], tables["A"]
 p_hat = np.asarray(p_hat, dtype=float)
 p_hat = p_hat / np.sum(p_hat)
 n_noise = len(p_hat)

 rng = np.random.default_rng(int(seed))
 Q = np.zeros((S, A), dtype=float)
 V = np.max(Q, axis=1)
 N_visits = np.zeros((S, A), dtype=float)
 total_pairs = S * A
 flat_pairs = np.arange(total_pairs, dtype=int)

 def update_pair(flat_idx):
  s_idx = int(flat_idx // A)
  a_idx = int(flat_idx % A)
  e_idx = int(rng.choice(n_noise, p=p_hat))
  target = compute_single_robust_target(
   Q,
   s_idx,
   a_idx,
   tables,
   robust_m,
   q_norm,
   p_hat,
   lambda_grid,
   discount,
   e_idx,
   V_next=V,
  )
  lr = (N_visits[s_idx, a_idx] + 1.0) ** (-float(w_lr))
  Q[s_idx, a_idx] += lr * (target - Q[s_idx, a_idx])
  V[s_idx] = np.max(Q[s_idx])
  N_visits[s_idx, a_idx] += 1.0

 for _ in range(int(coverage_passes)):
  rng.shuffle(flat_pairs)
  for flat_idx in flat_pairs:
   update_pair(flat_idx)

 updates_done = 0
 permutation_passes = 0
 num_updates = int(num_updates)
 while updates_done < num_updates:
  rng.shuffle(flat_pairs)
  permutation_passes += 1
  for flat_idx in flat_pairs:
   update_pair(flat_idx)
   updates_done += 1

   should_cover = (
    interleaved_coverage_passes > 0
    and interleaved_coverage_interval > 0
    and updates_done < num_updates
    and updates_done % int(interleaved_coverage_interval) == 0
   )
   if should_cover:
    for _ in range(int(interleaved_coverage_passes)):
     rng.shuffle(flat_pairs)
     for cov_idx in flat_pairs:
      update_pair(cov_idx)

   if updates_done >= num_updates:
    break

 bellman_target = compute_robust_bellman_target(
  Q, tables, robust_m, q_norm, p_hat, lambda_grid, discount
 )
 return {
  "Q": Q,
  "N_visits": N_visits,
  "coverage_pct": float(np.mean(N_visits > 0.0) * 100.0),
  "min_visits": float(np.min(N_visits)),
  "mean_visits": float(np.mean(N_visits)),
  "permutation_passes": int(permutation_passes),
  "bellman_residual_inf": float(np.max(np.abs(bellman_target - Q))),
 }


def evaluate_profile(tables, q_values, p_values, m_values, example, discount):
 costs = np.zeros((len(p_values), len(m_values)), dtype=float)
 for p_idx, p_value in enumerate(p_values):
  p_true = make_true_law(example, p_value)
  for m_idx in range(len(m_values)):
   costs[p_idx, m_idx] = evaluate_policy_exact(
    tables, q_values[m_idx], p_true, discount
   )
 return costs


def discounted_occupancy(tables, Q_ref, p_true, discount):
 S = tables["S"]
 p_true = np.asarray(p_true, dtype=float)
 p_true = p_true / np.sum(p_true)
 actions = np.argmax(Q_ref, axis=1)
 next_idx = tables["transition_table"][np.arange(S), actions, :]
 P = np.zeros((S, S), dtype=float)
 for e_idx, prob in enumerate(p_true):
  P[np.arange(S), next_idx[:, e_idx]] += prob
 init = np.ones(S, dtype=float) / S
 occ = np.linalg.solve(np.eye(S) - discount * P.T, init)
 occ = (1.0 - discount) * occ
 occ = np.maximum(occ, 0.0)
 if np.sum(occ) > 0:
  occ = occ / np.sum(occ)
 return occ


def policy_margin_and_disagreement(tables, Q_ref, Q_sampled, p_true, discount):
 ref_actions = np.argmax(Q_ref, axis=1)
 sampled_actions = np.argmax(Q_sampled, axis=1)
 sorted_q = np.sort(Q_ref, axis=1)
 margins = sorted_q[:, -1] - sorted_q[:, -2] if Q_ref.shape[1] > 1 else np.full(Q_ref.shape[0], np.inf)
 disagree = (ref_actions != sampled_actions).astype(float)
 occ = discounted_occupancy(tables, Q_ref, p_true, discount)
 return {
  "uniform_disagreement": float(np.mean(disagree)),
  "occupancy_disagreement": float(np.dot(occ, disagree)),
  "margin_min": float(np.min(margins)),
  "margin_median": float(np.median(margins)),
  "margin_mean": float(np.mean(margins)),
  "margin_occ_mean": float(np.dot(occ, margins)),
  "margin_share_lt_1e_4": float(np.mean(margins < 1e-4)),
  "margin_share_lt_1e_3": float(np.mean(margins < 1e-3)),
  "margin_share_lt_1e_2": float(np.mean(margins < 1e-2)),
 }


def interpolate_to_grid(p_src, m_src, values, p_dst, m_dst):
 p_src = np.asarray(p_src, dtype=float)
 m_src = np.asarray(m_src, dtype=float)
 values = np.asarray(values, dtype=float)
 p_dst = np.asarray(p_dst, dtype=float)
 m_dst = np.asarray(m_dst, dtype=float)
 by_m = np.vstack([np.interp(m_dst, m_src, row) for row in values])
 out = np.empty((len(p_dst), len(m_dst)), dtype=float)
 for j in range(len(m_dst)):
  out[:, j] = np.interp(p_dst, p_src, by_m[:, j])
 return out


def bootstrap_ci(values, rng_seed=123, n_boot=2000, alpha=0.05):
 values = np.asarray(values, dtype=float)
 values = values[np.isfinite(values)]
 if values.size == 0:
  return np.nan, np.nan, np.nan
 if values.size == 1:
  return float(values[0]), float(values[0]), float(values[0])
 rng = np.random.default_rng(rng_seed)
 boot = rng.choice(values, size=(int(n_boot), values.size), replace=True).mean(axis=1)
 return (
  float(np.mean(values)),
  float(np.quantile(boot, alpha / 2.0)),
  float(np.quantile(boot, 1.0 - alpha / 2.0)),
 )


class Timer:
 def __init__(self):
  self.start = time.time()

 def seconds(self):
  return float(time.time() - self.start)
