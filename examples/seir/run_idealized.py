import os
import sys
import time
import argparse
import numpy as np
_MPL_CACHE = os.path.join(os.environ.get("TMPDIR", "/tmp"), "robust_mfc_matplotlib")
os.makedirs(_MPL_CACHE, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _MPL_CACHE)
os.environ.setdefault("XDG_CACHE_HOME", _MPL_CACHE)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from engine.table_builder import precompute_tables
from engine.robust_solvers import compute_robust_bellman_target, make_lambda_grid
from engine.evaluation import evaluate_policy_exact
from engine.logger import TeeLogger, Timer
from seir_env import SEIRModularEnv

# ============================================================
# Configurations
# ============================================================
CONFIGS = {
 "mini": {
  "description": "Fast smoke test config",
  "n_disc": 8,
  "a_disc": 3,
 },
 "normal": {
  "description": "Full experimental sweep",
  "n_disc": 12,
  "a_disc": 10,
 }
}
P_HAT_DEFAULT = np.array([0.1, 0.8, 0.1])
PROFILE_M_FINE = np.array([0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08,
       0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0])


def parse_grid_arg(value, default):
 if not value:
  return np.asarray(default, dtype=float)
 return np.asarray([float(x.strip()) for x in value.split(",") if x.strip()], dtype=float)


def make_lambda_grid_from_arg(value):
 value = (value or "default").strip().lower()
 if value in ("", "default"):
  return make_lambda_grid()
 if value == "dense":
  linear = np.linspace(0.0, 150.0, 76)
  log = np.geomspace(0.25, 3000.0, 60)
  return np.unique(np.concatenate(([0.0], linear, log)))
 if value.startswith("max:"):
  return make_lambda_grid(float(value.split(":", 1)[1]))
 return np.unique(parse_grid_arg(value, []))

# ============================================================
# Idealized Value Iteration
# ============================================================
def run_value_iteration(tables, env, robust_m, discount, q_norm, p_hat, lambda_grid, max_iter=200, tol=1e-4):
 Q = np.zeros((tables["S"], tables["A"]))
 for i in range(max_iter):
  Q_new = compute_robust_bellman_target(
   Q, tables, robust_m, q_norm, p_hat, lambda_grid, discount
  )
  diff = np.max(np.abs(Q_new - Q))
  Q = Q_new
  if diff < tol:
   print(f"  Converged in {i+1} iterations (diff={diff:.6f}).")
   break
 else:
  print(f"  Warning: Did not converge after {max_iter} iterations.")
 return Q

def run_sweep(env, p_hat, betas, sweep_m, sweep_p, discount, q_norm, config, tables, lambda_grid):
 q_tables = {}
 results = np.zeros((len(sweep_p), len(sweep_m)))

 for m_idx, m_val in enumerate(sweep_m):
  print(f"\n--- Solving for M = {m_val:.3f} ---")
  Q_star = run_value_iteration(
   tables, env, m_val, discount, q_norm, p_hat, lambda_grid
  )
  q_tables[m_val] = Q_star

 print("\n--- Evaluating Policies under True Drift ---")
 for p_idx, p_val in enumerate(sweep_p):
  p_true = (1 - p_val) * p_hat + p_val * np.array([0.0, 0.0, 1.0])
  print(f" True drift p={p_val:.2f}: ", end="", flush=True)
  for m_idx, m_val in enumerate(sweep_m):
   cost = evaluate_policy_exact(
    tables, q_tables[m_val], p_true, discount,
   )
   results[p_idx, m_idx] = cost
   print(f"M={m_val:.3f}({cost:.3f}) ", end="", flush=True)
  print("")

 return q_tables, results

# ============================================================
# Plotting
# ============================================================
def plot_trajectories(env, tables, Q_m, p_true, T_max, output_dir):
 fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
 
 mu_inits = [
  np.array([0.99, 0.01, 0.00, 0.00]),
  np.array([0.80, 0.10, 0.10, 0.00]),
  np.array([0.50, 0.00, 0.50, 0.00])
 ]
 titles = [
  "Init: 1% Exposed",
  "Init: 10% Exp, 10% Inf",
  "Init: 50% Infected"
 ]
 
 N_E = len(p_true)
 n_trajectories = 5
 colors = plt.cm.tab10(np.linspace(0, 1, n_trajectories))
 
 for ax_idx, mu0 in enumerate(mu_inits):
  ax = axes[ax_idx]
  for traj_idx in range(n_trajectories):
   np.random.seed(42 + traj_idx + ax_idx * 100)
   _, start_idx = tables["grid_tree"].query(mu0)
   s_idx = int(start_idx)
   traj_I = [tables["states"][s_idx][2]]
   
   for t in range(T_max):
    a_idx = np.argmax(Q_m[s_idx])
    e_realized = np.random.choice(N_E, p=p_true)
    s_idx = tables["transition_table"][s_idx, a_idx, e_realized]
    traj_I.append(tables["states"][s_idx][2])
    
   ax.plot(range(T_max + 1), traj_I, marker='o', markersize=3, 
     color=colors[traj_idx], alpha=0.7, 
     label=f'Traj {traj_idx+1}' if ax_idx==0 else None)
   
  ax.set_title(titles[ax_idx], fontsize=12)
  ax.set_xlabel('Time step', fontsize=11)
  if ax_idx == 0:
   ax.set_ylabel('Fraction Infected', fontsize=11)
  ax.grid(True, ls='--', alpha=0.5)
  
 fig.suptitle(f'{env.name} — Distribution Trajectories (M=0.3, True p=0.0)', fontsize=14)
 fig.tight_layout()
 fig.savefig(os.path.join(output_dir, "plot3_trajectories.pdf"))
 plt.close(fig)

def make_plots(env, tables, q_tables, sweep_m, sweep_p, results_mean, output_dir):
 fig, ax = plt.subplots(figsize=(8, 6))
 colors = plt.cm.viridis(np.linspace(0, 1, len(sweep_p)))
 for p_idx, p_val in enumerate(sweep_p):
  ax.plot(sweep_m, results_mean[p_idx, :], marker='o',
    color=colors[p_idx], label=f'True p={p_val:.2f}')
 ax.set_xlabel('Robustness radius M', fontsize=12)
 ax.set_ylabel('Expected Discounted Cost', fontsize=12)
 ax.set_title(f'{env.name} — Robustness Profiles', fontsize=14)
 ax.legend(title="True drift probability", bbox_to_anchor=(1.05, 1), loc='upper left')
 ax.grid(True, ls='--', alpha=0.5)
 fig.tight_layout()
 fig.savefig(os.path.join(output_dir, "plot1_1d_profiles.png"))
 plt.close(fig)

 fig, ax = plt.subplots(figsize=(8, 6))
 im = ax.imshow(results_mean, origin='lower', aspect='auto', cmap='plasma',
     extent=[sweep_m[0], sweep_m[-1], sweep_p[0], sweep_p[-1]])
 ax.set_xlabel('Robustness radius M', fontsize=12)
 ax.set_ylabel('True drift probability p', fontsize=12)
 ax.set_title(f'{env.name} — Cost Landscape', fontsize=14)
 fig.colorbar(im, ax=ax, label='Expected Discounted Cost')
 fig.tight_layout()
 fig.savefig(os.path.join(output_dir, "plot2_heatmap.png"))
 plt.close(fig)
 
 if 0.3 in q_tables:
  plot_trajectories(env, tables, q_tables[0.3], p_true=P_HAT_DEFAULT, T_max=20, output_dir=output_dir)

 print(f"\nPlots saved to {output_dir}/")

# ============================================================
# Main
# ============================================================
def main():
 parser = argparse.ArgumentParser()
 parser.add_argument("--mini", action="store_true")
 parser.add_argument("--fine", action="store_true")
 parser.add_argument("--discount", type=float, default=0.9)
 parser.add_argument("--q-norm", type=int, default=1)
 parser.add_argument("--cost-dist", type=float, default=0.85)
 parser.add_argument("--dist-eff", type=float, default=1.0)
 parser.add_argument("--c-f", type=float, default=3.2)
 parser.add_argument("--betas", type=str, default="0.5,0.81,1.8")
 parser.add_argument("--a-disc", type=int, default=0)
 parser.add_argument("--n-disc", type=int, default=0)
 parser.add_argument("--m-grid", type=str, default="")
 parser.add_argument("--p-grid", type=str, default="")
 parser.add_argument("--seed", type=int, default=0)
 parser.add_argument("--output-root", type=str, default="")
 parser.add_argument("--run-id", type=str, default="")
 parser.add_argument("--save-q", action="store_true")
 parser.add_argument("--lambda-grid", type=str, default="default")
 parser.add_argument("--suffix", type=str, default="")
 args = parser.parse_args()

 mode = "mini" if args.mini else ("fine" if args.fine else "normal")
 config = dict(CONFIGS["normal"] if mode == "fine" else CONFIGS[mode])
 if args.a_disc > 0:
  config["a_disc"] = args.a_disc
 if args.n_disc > 0:
  config["n_disc"] = args.n_disc
 betas_array = np.array([float(x) for x in args.betas.split(",")])

 env = SEIRModularEnv(
  cost_distancing=args.cost_dist,
  distancing_efficacy=args.dist_eff,
  lipschitz_constant=args.c_f,
 )
 discount = args.discount
 q_norm = args.q_norm
 p_hat = P_HAT_DEFAULT.copy()

 if mode == "mini":
  sweep_m = np.array([0.0, 0.02, 0.05, 0.3, 1.0])
  sweep_p = np.array([0.0, 0.3, 0.6, 1.0])
 elif mode == "fine":
  sweep_m = PROFILE_M_FINE
  sweep_p = np.linspace(0.0, 1.0, 10)
 else:
  sweep_m = PROFILE_M_FINE
  sweep_p = np.linspace(0.0, 1.0, 11)
 sweep_m = parse_grid_arg(args.m_grid, sweep_m)
 sweep_p = parse_grid_arg(args.p_grid, sweep_p)
 lambda_grid = make_lambda_grid_from_arg(args.lambda_grid)

 timestamp = time.strftime("%Y%m%d-%H%M%S")
 suffix = f"-{args.suffix}" if args.suffix else ""
 if args.output_root:
  output_dir = os.path.join(args.output_root, args.run_id or f"outputs-idealized-SEIR-{mode}{suffix}-{timestamp}")
 else:
  output_dir = os.path.join(
   os.path.dirname(os.path.abspath(__file__)),
   f"outputs-idealized-SEIR-{mode}{suffix}-{timestamp}",
  )
 os.makedirs(output_dir, exist_ok=True)

 logger = TeeLogger(os.path.join(output_dir, "run.log"))
 sys.stdout = logger
 timer = Timer()

 print(f"{'='*60}")
 print(f"Robust MFC Q-Learning — SEIR Idealized Solver (MODULAR)")
 print(f"Config: {mode}")
 print(f"Lambda grid: {args.lambda_grid} ({len(lambda_grid)} points)")
 print(f"{'='*60}")

 tables = precompute_tables(
  env, n_disc=config["n_disc"], a_disc=config["a_disc"], noise_grid=betas_array
 )

 q_tables, results_mean = run_sweep(
  env, p_hat, betas_array, sweep_m, sweep_p, discount, q_norm, config, tables, lambda_grid
 )

 make_plots(env, tables, q_tables, sweep_m, sweep_p, results_mean, output_dir)
 np.savez(os.path.join(output_dir, "sweep_results.npz"),
    results_matrix=results_mean, p_values=sweep_p, m_values=sweep_m,
    p_hat=p_hat, betas=betas_array, discount=discount, q_norm=q_norm,
    cost_distancing=args.cost_dist, dist_eff=args.dist_eff,
    lipschitz_constant=args.c_f, n_disc=config["n_disc"],
    a_disc=config["a_disc"])
 if args.save_q:
  q_values = np.asarray([q_tables[m] for m in sweep_m], dtype=float)
  np.savez_compressed(os.path.join(output_dir, "q_tables.npz"),
       m_values=sweep_m, q_values=q_values)

 print(f"{timer()} Done! All outputs in {output_dir}")

if __name__ == "__main__":
 main()
