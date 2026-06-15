#!/usr/bin/env python3
"""Fast confidence checks for the robust MFC Q-learning codebase.

Runs on a laptop in under a minute with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1.
Exits 0 on success, nonzero on any failure.
"""
import os
import sys

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
 sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import traceback
import numpy as np


def header(msg):
 print(f"\n{'=' * 60}")
 print(f" {msg}")
 print(f"{'=' * 60}")


def check(condition, label):
 if not condition:
  raise AssertionError(f"FAIL: {label}")
 print(f" OK: {label}")


# ---------------------------------------------------------------------------
# 1. Import all modules
# ---------------------------------------------------------------------------
header("1. Module imports")

import engine
from engine.mfc_env import MFCEnvironment
from engine.mfc_utils import (
 generate_simplex_states,
 precompute_policies,
 project_to_simplex_grid,
 compute_W1_1d,
 compute_wasserstein_power_1d,
 compute_W1_matrix,
 compute_wasserstein_power_matrix,
 compute_noise_cost_matrix,
)
from engine.table_builder import precompute_tables
from engine.robust_solvers import (
 make_lambda_grid,
 compute_robust_bellman_target,
 compute_single_robust_target,
 run_async_qlearning,
)
from engine.evaluation import evaluate_policy_exact, evaluate_policy_vectorized
from engine.convergence import (
 make_log_checkpoints,
 run_value_iteration_trace,
 compute_policy_costs,
 run_sampled_qlearning_trace,
)
from engine.logger import Timer, TeeLogger

from examples.systemic_risk.sysrisk_env import SysRiskModularEnv
from examples.sis.sis_env import SISModularEnv
from examples.seir.seir_env import SEIRModularEnv

print(" All imports successful.")

# ---------------------------------------------------------------------------
# 2. Build tiny tables for all three environments
# ---------------------------------------------------------------------------
header("2. Table precomputation (tiny grids)")

DISCOUNT = 0.5
Q_NORM = 1

_sys_env = SysRiskModularEnv()
_sys_p_hat = np.array([0.1, 0.8, 0.1])
_sys_noise = np.array([-1.0, 0.0, 1.0])
_tbl_sys = precompute_tables(_sys_env, n_disc=2, a_disc=2, noise_grid=_sys_noise)
check(_tbl_sys["S"] > 0, "Systemic Risk table built (S>0)")
check(_tbl_sys["A"] > 0, "Systemic Risk table built (A>0)")

_sis_env = SISModularEnv()
_sis_noise = np.array([1.0, 2.0, 3.0])
_tbl_sis = precompute_tables(_sis_env, n_disc=2, a_disc=2, noise_grid=_sis_noise)
check(_tbl_sis["S"] > 0, "SIS table built (S>0)")
check(_tbl_sis["A"] > 0, "SIS table built (A>0)")

_seir_env = SEIRModularEnv()
_seir_noise = np.array([1.0, 2.0, 3.0])
_tbl_seir = precompute_tables(_seir_env, n_disc=2, a_disc=2, noise_grid=_seir_noise)
check(_tbl_seir["S"] > 0, "SEIR table built (S>0)")
check(_tbl_seir["A"] > 0, "SEIR table built (A>0)")


# ---------------------------------------------------------------------------
# 3. Robust dual sanity properties (Systemic Risk, tiny table)
# ---------------------------------------------------------------------------
header("3. Robust dual sanity properties")

lam_grid = make_lambda_grid()

# 3a. At m=0, idealized target equals nominal expectation
Q_zero = np.zeros((_tbl_sys["S"], _tbl_sys["A"]))
target_m0 = compute_robust_bellman_target(
 Q_zero, _tbl_sys, robust_m=0.0, q_norm=Q_NORM,
 p_hat=_sys_p_hat, lambda_grid=lam_grid, discount=DISCOUNT,
)
nominal_exp = _tbl_sys["reward_table"] + DISCOUNT * np.sum(
 _sys_p_hat * np.zeros((_tbl_sys["S"], _tbl_sys["A"], len(_sys_p_hat))), axis=-1
)
check(np.allclose(target_m0, nominal_exp), "m=0 idealized target matches nominal expectation")

# 3b. At m=0, sampled target equals one-sample target
_s_idx, _a_idx = 0, 0
V0 = np.max(Q_zero, axis=1)
target_sampled_m0 = compute_single_robust_target(
 Q_zero, _s_idx, _a_idx, _tbl_sys, robust_m=0.0, q_norm=Q_NORM,
 p_hat=_sys_p_hat, lambda_grid=lam_grid, discount=DISCOUNT,
 e_realized_idx=1, V_next=V0,
)
check(np.isfinite(target_sampled_m0), "m=0 sampled target is finite")
check(
 abs(target_sampled_m0 - _tbl_sys["reward_table"][_s_idx, _a_idx]) < 1e-12,
 "m=0 sampled target matches reward (V=0)",
)

# 3c. Constant continuation values remain unchanged across robustness radii
V_const = np.ones(_tbl_sys["S"]) * 42.0
Q_const = np.tile(V_const[:, None], (1, _tbl_sys["A"]))
target_m0_const = compute_robust_bellman_target(
 Q_const, _tbl_sys, robust_m=0.0, q_norm=Q_NORM,
 p_hat=_sys_p_hat, lambda_grid=lam_grid, discount=DISCOUNT,
)
target_m5_const = compute_robust_bellman_target(
 Q_const, _tbl_sys, robust_m=0.5, q_norm=Q_NORM,
 p_hat=_sys_p_hat, lambda_grid=lam_grid, discount=DISCOUNT,
)
# For constant V=42, V_next is all 42, and the robust dual collapses to 42
# (inner min over noise picks 42, and the phi reduces to 42 for both robust and non-robust)
check(
 np.allclose(target_m0_const, target_m5_const),
 "Constant continuation values unchanged across M (0 vs 0.5)",
)

# 3d. Robust target increases discount * penalty relative to nominal when V varies
Q_rand = np.random.default_rng(123).uniform(-1, 1, (_tbl_sys["S"], _tbl_sys["A"]))
tgt_nom = compute_robust_bellman_target(
 Q_rand, _tbl_sys, robust_m=0.0, q_norm=Q_NORM,
 p_hat=_sys_p_hat, lambda_grid=lam_grid, discount=DISCOUNT,
)
tgt_rob = compute_robust_bellman_target(
 Q_rand, _tbl_sys, robust_m=0.5, q_norm=Q_NORM,
 p_hat=_sys_p_hat, lambda_grid=lam_grid, discount=DISCOUNT,
)
# For m>0 there is an additional penalty term; the robust target can be above or below
# nominal depending on the value landscape, but they should differ and both be finite
check(np.all(np.isfinite(tgt_nom)), "Nominal target finite")
check(np.all(np.isfinite(tgt_rob)), "Robust target finite")
check(not np.allclose(tgt_nom, tgt_rob), "Robust target differs from nominal (non-constant V)")


# ---------------------------------------------------------------------------
# 4. Tiny idealized Bellman iteration
# ---------------------------------------------------------------------------
header("4. Idealized Bellman iteration (tiny)")

for label, tbl, p_hat in [
 ("Systemic Risk", _tbl_sys, _sys_p_hat),
 ("SIS", _tbl_sis, np.ones(len(_sis_noise)) / len(_sis_noise)),
 ("SEIR", _tbl_seir, np.ones(len(_seir_noise)) / len(_seir_noise)),
]:
 trace = run_value_iteration_trace(
  tbl, robust_m=0.3, discount=DISCOUNT, q_norm=Q_NORM,
  p_hat=p_hat, lambda_grid=lam_grid, max_iter=50, tol=1e-6,
 )
 Q_ref = trace["Q_ref"]
 check(np.all(np.isfinite(Q_ref)), f"{label}: Q_ref finite")
 check(np.max(np.abs(Q_ref)) > 0, f"{label}: non-zero Q_ref (max={np.max(np.abs(Q_ref)):.4g})")
 print(f" {label}: {len(trace['iterations'])} iters, residual={trace['bellman_residual_inf'][-1]:.3e}")


# ---------------------------------------------------------------------------
# 5. Tiny asynchronous updates
# ---------------------------------------------------------------------------
header("5. Asynchronous Q-learning (tiny)")

for label, tbl, p_hat, env_obj in [
 ("Systemic Risk", _tbl_sys, _sys_p_hat, _sys_env),
 ("SIS", _tbl_sis, np.ones(len(_sis_noise)) / len(_sis_noise), _sis_env),
 ("SEIR", _tbl_seir, np.ones(len(_seir_noise)) / len(_seir_noise), _seir_env),
]:
 S, A = tbl["S"], tbl["A"]
 total_pairs = S * A
 num_updates = total_pairs * 5

 Q_async = run_async_qlearning(
  tbl, robust_m=0.3, discount=DISCOUNT, q_norm=Q_NORM,
  p_hat=p_hat, lambda_grid=lam_grid,
  num_updates=num_updates, w_lr=0.7, coverage_passes=1,
 )
 check(np.all(np.isfinite(Q_async)), f"{label}: async Q finite")
 coverage = np.mean(np.max(np.abs(Q_async), axis=1) != 0) * 100
 check(coverage > 0, f"{label}: positive coverage ({coverage:.1f}%)")
 print(f" {label}: {num_updates} updates, coverage={coverage:.1f}%")


# ---------------------------------------------------------------------------
# 6. Evaluation sanity
# ---------------------------------------------------------------------------
header("6. Policy evaluation sanity")

for label, tbl, p_hat in [
 ("Systemic Risk", _tbl_sys, _sys_p_hat),
 ("SIS", _tbl_sis, np.ones(len(_sis_noise)) / len(_sis_noise)),
 ("SEIR", _tbl_seir, np.ones(len(_seir_noise)) / len(_seir_noise)),
]:
 Q_test = np.zeros((tbl["S"], tbl["A"]))
 cost_exact = evaluate_policy_exact(tbl, Q_test, p_hat, DISCOUNT)
 check(np.isfinite(cost_exact) and cost_exact >= 0.0, f"{label}: exact eval finite and non-negative ({cost_exact:.4g})")

 cost_sim = evaluate_policy_vectorized(tbl, Q_test, p_hat, DISCOUNT, n_sims=100, T_max=20, seed=42)
 check(np.isfinite(cost_sim) and cost_sim >= 0.0, f"{label}: sim eval finite and non-negative ({cost_sim:.4g})")


# ---------------------------------------------------------------------------
# 7. Utility function sanity
# ---------------------------------------------------------------------------
header("7. Utility functions")

states = np.array(list(generate_simplex_states(3, 3)), dtype=float) / 3.0
check(len(states) > 0, "simplex_states non-empty")
check(np.allclose(states.sum(axis=1), 1.0), "simplex states sum to 1")

local_actions, global_policies = precompute_policies(3, 2, 3)
check(len(local_actions) > 0, "local_actions non-empty")
check(global_policies.shape[1] == 3, "global_policies shape correct")

W1 = compute_W1_1d(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
check(abs(W1 - 1.0) < 1e-12, f"W1([1,0],[0,1]) == 1 (got {W1})")

W1_mat = compute_W1_matrix(states)
check(W1_mat.shape == (len(states), len(states)), "W1 matrix shape correct")
check(np.all(np.diag(W1_mat) == 0.0), "W1 matrix diagonal zero")
check(np.allclose(W1_mat, W1_mat.T), "W1 matrix symmetric")

Wq_mat = compute_wasserstein_power_matrix(states, q_norm=2)
check(Wq_mat.shape == (len(states), len(states)), "W^2 matrix shape correct")

noise_grid = np.array([-1.0, 0.0, 1.0])
nc_mat = compute_noise_cost_matrix(noise_grid, q_norm=1)
check(nc_mat.shape == (3, 3), "noise cost matrix shape")
check(np.all(np.diag(nc_mat) == 0.0), "noise cost matrix diagonal zero")

lg = make_lambda_grid()
check(len(lg) > 2, f"lambda grid has > 2 entries ({len(lg)})")
check(lg[0] == 0.0, "lambda grid starts at 0")

cpts = make_log_checkpoints(total_pairs=10, num_updates=200, n_points=20)
check(len(cpts) > 0, "log checkpoints non-empty")
check(cpts[-1] == 200, "log checkpoints end at num_updates")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
header("SANITY CHECK SUMMARY")
print("\n All checks passed.")
print(f" Python: {sys.version}")
print(f" NumPy: {np.__version__}")
print(f" Scipy: ", end="")
try:
 import scipy
 print(scipy.__version__)
except Exception:
 print("(not available)")
print(f" Matplotlib: ", end="")
try:
 import matplotlib
 print(matplotlib.__version__)
except Exception:
 print("(not available)")

print("\n Ready for publication release preparation.\n")
sys.exit(0)
