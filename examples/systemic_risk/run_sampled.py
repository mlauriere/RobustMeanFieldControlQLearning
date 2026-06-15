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
from engine.robust_solvers import run_async_qlearning, make_lambda_grid
from engine.evaluation import evaluate_policy_exact
from engine.logger import TeeLogger, Timer
from sysrisk_env import SysRiskModularEnv

# ============================================================
# Configurations
# ============================================================
CONFIGS = {
 "mini": {
  "description": "Fast smoke test config",
  "n_disc": 2,
  "a_disc": 2,
 },
 "normal": {
  "description": "Full experimental sweep",
  "n_disc": 5,
  "a_disc": 2,
 }
}
P_HAT_DEFAULT = np.array([0.1, 0.8, 0.1])
E_VALS = np.array([-1, 0, 1])


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
# Sweep Logic
# ============================================================
def run_sweep(env, p_hat, sweep_m, sweep_p, discount, q_norm, num_updates, num_seeds, config, tables, coverage_passes, w_lr, lambda_grid, seed_base):
 q_tables = {m: [] for m in sweep_m}
 results = np.zeros((num_seeds, len(sweep_p), len(sweep_m)))

 for seed in range(num_seeds):
  np.random.seed(seed_base + seed)
  print(f"\n{'*'*20} SEED {seed+1}/{num_seeds} {'*'*20}")
  
  for m_idx, m_val in enumerate(sweep_m):
   print(f"\n--- Solving for M = {m_val:.2f} ---")
   Q_star = run_async_qlearning(
    tables, m_val, discount, q_norm, p_hat, lambda_grid,
    num_updates=num_updates, w_lr=w_lr,
    coverage_passes=coverage_passes,
   )
   q_tables[m_val].append(Q_star)

  print("\n--- Evaluating Policies under True Drift ---")
  for p_idx, p_val in enumerate(sweep_p):
   p_true = (1.0 - p_val) * p_hat + p_val * np.array([1.0, 0.0, 0.0])
   print(f" True drift p={p_val:.2f}: ", end="", flush=True)
   for m_idx, m_val in enumerate(sweep_m):
    cost = evaluate_policy_exact(
     tables, q_tables[m_val][-1], p_true, discount,
    )
    results[seed, p_idx, m_idx] = cost
    print(f"M={m_val:.1f}({cost:.3f}) ", end="", flush=True)
   print("")

 return q_tables, results

# ============================================================
# Plotting
# ============================================================
def make_plots(env, tables, q_tables, sweep_m, sweep_p, results, output_dir):
 results_mean = np.mean(results, axis=0)
 results_std = np.std(results, axis=0)

 fig, ax = plt.subplots(figsize=(8, 6))
 colors = plt.cm.viridis(np.linspace(0, 1, len(sweep_p)))
 for p_idx, p_val in enumerate(sweep_p):
  ax.errorbar(sweep_m, results_mean[p_idx, :], yerr=results_std[p_idx, :], marker='o',
     color=colors[p_idx], label=f'True p={p_val:.2f}', capsize=4)
 ax.set_xlabel('Robustness radius M', fontsize=12)
 ax.set_ylabel('Expected Discounted Cost', fontsize=12)
 ax.set_title(f'{env.name} — Robustness Profiles (Sampled)', fontsize=14)
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
 ax.set_title(f'{env.name} — Cost Landscape (Sampled)', fontsize=14)
 fig.colorbar(im, ax=ax, label='Expected Discounted Cost')
 fig.tight_layout()
 fig.savefig(os.path.join(output_dir, "plot2_heatmap.png"))
 plt.close(fig)

 print(f"\nPlots saved to {output_dir}/")

# ============================================================
# Main
# ============================================================
def main():
 parser = argparse.ArgumentParser()
 parser.add_argument("--mini", action="store_true")
 parser.add_argument("--num-updates", type=int, default=0)
 parser.add_argument("--num-episodes", type=int, default=0)
 parser.add_argument("--t-episode", type=int, default=10)
 parser.add_argument("--num-seeds", type=int, default=1)
 parser.add_argument("--discount", type=float, default=0.5)
 parser.add_argument("--q-norm", type=int, default=1)
 parser.add_argument("--cliff", type=float, default=2.0)
 parser.add_argument("--coverage-passes", type=int, default=1)
 parser.add_argument("--w-lr", type=float, default=0.7)
 parser.add_argument("--m-grid", type=str, default="")
 parser.add_argument("--p-grid", type=str, default="")
 parser.add_argument("--seed", type=int, default=42)
 parser.add_argument("--output-root", type=str, default="")
 parser.add_argument("--run-id", type=str, default="")
 parser.add_argument("--save-q", action="store_true")
 parser.add_argument("--lambda-grid", type=str, default="default")
 parser.add_argument("--suffix", type=str, default="")
 args = parser.parse_args()

 mode = "mini" if args.mini else "normal"
 config = CONFIGS[mode]

 if args.num_updates > 0:
  num_updates = args.num_updates
 else:
  num_episodes = args.num_episodes if args.num_episodes > 0 else (1000 if mode == "mini" else 5000)
  num_updates = num_episodes * args.t_episode

 env = SysRiskModularEnv(cliff_penalty=args.cliff)
 discount = args.discount
 q_norm = args.q_norm
 p_hat = P_HAT_DEFAULT.copy()

 if mode == "mini":
  sweep_m = np.array([0.0, 0.3, 0.6, 1.0])
  sweep_p = np.array([0.0, 0.3, 0.6, 1.0])
 else:
  sweep_m = np.array([0.0, 0.3, 0.6, 1.0])
  sweep_p = np.linspace(0.0, 1.0, 5)
 sweep_m = parse_grid_arg(args.m_grid, sweep_m)
 sweep_p = parse_grid_arg(args.p_grid, sweep_p)
 lambda_grid = make_lambda_grid_from_arg(args.lambda_grid)

 timestamp = time.strftime("%Y%m%d-%H%M%S")
 suffix = f"-{args.suffix}" if args.suffix else ""
 if args.output_root:
  output_dir = os.path.join(args.output_root, args.run_id or f"outputs-async-SystemicRisk-{mode}{suffix}-{timestamp}")
 else:
  output_dir = os.path.join(
   os.path.dirname(os.path.abspath(__file__)),
   f"outputs-async-SystemicRisk-{mode}{suffix}-{timestamp}",
  )
 os.makedirs(output_dir, exist_ok=True)

 logger = TeeLogger(os.path.join(output_dir, "run.log"))
 sys.stdout = logger
 timer = Timer()

 print(f"{'='*60}")
 print(f"Robust MFC Q-Learning — Systemic Risk Async Solver (MODULAR)")
 print(f"Config: {mode}")
 print(f"Updates: {num_updates}")
 print(f"Learning-rate exponent w: {args.w_lr}")
 print(f"Seed base: {args.seed}")
 print(f"Lambda grid: {args.lambda_grid} ({len(lambda_grid)} points)")
 if args.num_updates <= 0:
  print(f"Compatibility: updates = episodes * T_episode = {num_episodes} * {args.t_episode}")
 print(f"{'='*60}")

 tables = precompute_tables(
  env, n_disc=config["n_disc"], a_disc=config["a_disc"], noise_grid=E_VALS
 )

 q_tables, results = run_sweep(
  env, p_hat, sweep_m, sweep_p, discount, q_norm, num_updates,
  args.num_seeds, config, tables, args.coverage_passes, args.w_lr,
  lambda_grid, args.seed,
 )

 make_plots(env, tables, q_tables, sweep_m, sweep_p, results, output_dir)
 np.savez(os.path.join(output_dir, "sweep_results.npz"),
    results_matrix=results, p_values=sweep_p, m_values=sweep_m,
    p_hat=p_hat, discount=discount, q_norm=q_norm,
    num_updates=num_updates,
    coverage_passes=args.coverage_passes,
    w_lr=args.w_lr, seed_base=args.seed,
    n_disc=config["n_disc"], a_disc=config["a_disc"])
 if args.save_q:
  q_values = np.asarray([[q_tables[m][seed] for m in sweep_m] for seed in range(args.num_seeds)], dtype=float)
  np.savez_compressed(os.path.join(output_dir, "q_tables.npz"),
       m_values=sweep_m, q_values=q_values, seed_base=args.seed)

 print(f"{timer()} Done! All outputs in {output_dir}")

if __name__ == "__main__":
 main()
