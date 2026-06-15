import numpy as np

from .evaluation import evaluate_policy_exact
from .robust_solvers import (
 compute_robust_bellman_target,
 compute_single_robust_target,
)


def make_log_checkpoints(total_pairs: int, num_updates: int, n_points: int = 30) -> np.ndarray:
 """
 Builds log-spaced checkpoints from one full state-action permutation to the
 final update count. If the budget is smaller than one permutation, the
 final update is still recorded.
 """
 num_updates = int(num_updates)
 if num_updates <= 0:
  return np.array([], dtype=int)

 start = min(max(1, int(total_pairs)), num_updates)
 raw = np.geomspace(start, num_updates, num=max(2, int(n_points)))
 checkpoints = np.unique(np.rint(raw).astype(int))
 checkpoints = checkpoints[(checkpoints >= 1) & (checkpoints <= num_updates)]
 if checkpoints[-1] != num_updates:
  checkpoints = np.append(checkpoints, num_updates)
 return checkpoints.astype(int)


def run_value_iteration_trace(
 tables: dict,
 robust_m: float,
 discount: float,
 q_norm: int,
 p_hat: np.ndarray,
 lambda_grid: np.ndarray,
 max_iter: int = 800,
 tol: float = 1e-9,
) -> dict:
 """
 Runs deterministic Bellman iteration and returns residual and reference
 errors. The final iterate is treated as the finite-grid reference Q^*.
 """
 Q = np.zeros((tables["S"], tables["A"]), dtype=float)
 snapshots = []
 residuals = []

 for _ in range(int(max_iter)):
  Q_new = compute_robust_bellman_target(
   Q, tables, robust_m, q_norm, p_hat, lambda_grid, discount
  )
  residual = float(np.max(np.abs(Q_new - Q)))
  Q = Q_new
  snapshots.append(Q.copy())
  residuals.append(residual)
  if residual < tol:
   break

 Q_ref = Q.copy()
 q_sup_errors = np.array(
  [np.max(np.abs(Q_k - Q_ref)) for Q_k in snapshots], dtype=float
 )
 q_mae_errors = np.array(
  [np.mean(np.abs(Q_k - Q_ref)) for Q_k in snapshots], dtype=float
 )

 return {
  "Q_ref": Q_ref,
  "iterations": np.arange(1, len(residuals) + 1, dtype=int),
  "bellman_residual_inf": np.asarray(residuals, dtype=float),
  "q_sup_error": q_sup_errors,
  "q_mae_error": q_mae_errors,
  "converged": bool(residuals and residuals[-1] < tol),
 }


def compute_policy_costs(
 tables: dict,
 Q: np.ndarray,
 eval_laws: dict,
 discount: float,
) -> dict:
 """Evaluates one greedy policy under each named common-noise law."""
 return {
  name: evaluate_policy_exact(tables, Q, p_true, discount)
  for name, p_true in eval_laws.items()
 }


def run_sampled_qlearning_trace(
 tables: dict,
 robust_m: float,
 discount: float,
 q_norm: int,
 p_hat: np.ndarray,
 lambda_grid: np.ndarray,
 num_updates: int,
 checkpoints: np.ndarray,
 seed: int,
 Q_ref: np.ndarray,
 ref_costs: dict,
 eval_laws: dict,
 w_lr: float = 0.7,
) -> dict:
 """
 Runs the asynchronous sampled update and records checkpoint metrics.

 The update locations are repeated random permutations of all state-action
 pairs, independent of Q and independent of the common-noise samples.
 """
 S, A = tables["S"], tables["A"]
 p_hat = np.asarray(p_hat, dtype=float)
 p_hat = p_hat / np.sum(p_hat)
 n_noise = len(p_hat)

 rng = np.random.default_rng(int(seed))
 Q = np.zeros((S, A), dtype=float)
 V = np.max(Q, axis=1)
 N_visits = np.zeros((S, A), dtype=float)

 checkpoints = np.asarray(checkpoints, dtype=int)
 checkpoint_set = set(int(x) for x in checkpoints)
 total_pairs = S * A
 flat_pairs = np.arange(total_pairs, dtype=int)
 records = []

 def record(update_count: int):
  bellman_target = compute_robust_bellman_target(
   Q, tables, robust_m, q_norm, p_hat, lambda_grid, discount
  )
  costs = compute_policy_costs(tables, Q, eval_laws, discount)
  row = {
   "updates": int(update_count),
   "q_sup_error": float(np.max(np.abs(Q - Q_ref))),
   "q_mae_error": float(np.mean(np.abs(Q - Q_ref))),
   "bellman_residual_inf": float(np.max(np.abs(bellman_target - Q))),
   "coverage_pct": float(np.mean(N_visits > 0.0) * 100.0),
   "min_visits": float(np.min(N_visits)),
   "mean_visits": float(np.mean(N_visits)),
  }
  for name, cost in costs.items():
   gap = float(cost - ref_costs[name])
   row[f"policy_cost_{name}"] = float(cost)
   row[f"reference_cost_{name}"] = float(ref_costs[name])
   row[f"policy_gap_{name}"] = gap
   row[f"policy_gap_abs_{name}"] = abs(gap)
  records.append(row)

 updates_done = 0
 while updates_done < int(num_updates):
  rng.shuffle(flat_pairs)
  for flat_idx in flat_pairs:
   s_idx = int(flat_idx // A)
   a_idx = int(flat_idx % A)
   e_realized_idx = int(rng.choice(n_noise, p=p_hat))
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
    e_realized_idx,
    V_next=V,
   )
   lr = (N_visits[s_idx, a_idx] + 1.0) ** (-w_lr)
   Q[s_idx, a_idx] += lr * (target - Q[s_idx, a_idx])
   V[s_idx] = np.max(Q[s_idx])
   N_visits[s_idx, a_idx] += 1.0
   updates_done += 1

   if updates_done in checkpoint_set:
    record(updates_done)

   if updates_done >= int(num_updates):
    break

 return {
  "Q": Q,
  "N_visits": N_visits,
  "records": records,
 }
