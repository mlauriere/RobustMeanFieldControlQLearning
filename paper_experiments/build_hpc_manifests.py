#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

from campaign_lib import (
 CAMPAIGN_DIR,
 M_EPI,
 M_SYS,
 MANIFEST_FIELDS,
 P_COARSE,
 P_FINE,
 grid_to_str,
 write_csv,
)


def base_row(run_id, stage, task, example, n_disc, a_disc, discount, m_grid, p_grid, **kwargs):
 defaults = {
  "q_norm": 1,
  "num_updates": "",
  "seed": "",
  "w_lr": 0.7,
  "coverage_passes": 0,
  "interleaved_coverage_passes": 0,
  "interleaved_coverage_interval": 0,
  "n_checkpoints": 40,
  "idealized_max_iter": 1000,
  "idealized_tol": 1e-9,
  "lambda_grid": "default",
  "save_q": 1,
  "cliff": 2.0,
  "cost_dist": 0.5 if example == "sis" else 0.85,
  "dist_eff": 1.0,
  "c_f": 1.0 if example == "sysrisk" else 1.8 if example == "sis" else 3.2,
  "notes": "",
 }
 defaults.update(kwargs)
 row = {
  "run_id": run_id,
  "stage": stage,
  "task": task,
  "example": example,
  "n_disc": n_disc,
  "a_disc": a_disc,
  "discount": discount,
  "m_grid": grid_to_str(m_grid),
  "p_grid": grid_to_str(p_grid),
 }
 row.update(defaults)
 return row


def write_stage(path, rows):
 write_csv(path, MANIFEST_FIELDS, rows)
 return rows


def build_stage0():
 rows = []
 specs = {
  "sysrisk": (2, 2, 0.5, [0.0, 0.3], [0.0, 1.0], 3000),
  "sis": (8, 3, 0.5, [0.0, 0.3], [0.0, 1.0], 1000),
  "seir": (4, 3, 0.9, [0.0, 0.3], [0.0, 1.0], 2000),
 }
 for example, (n_disc, a_disc, discount, m_grid, p_grid, updates) in specs.items():
  rows.append(base_row(f"s0_{example}_ideal", "stage0_smoke", "idealized_profile", example, n_disc, a_disc, discount, m_grid, p_grid))
  rows.append(
   base_row(
    f"s0_{example}_sampled_seed00",
    "stage0_smoke",
    "sampled_profile",
    example,
    n_disc,
    a_disc,
    discount,
    m_grid,
    p_grid,
    num_updates=updates,
    seed=0,
   )
  )
  rows.append(
   base_row(
    f"s0_{example}_convergence_seed00",
    "stage0_smoke",
    "convergence",
    example,
    n_disc,
    a_disc,
    discount,
    [0.3],
    [0.0, 0.5, 1.0],
    num_updates=max(500, updates),
    seed=0,
    n_checkpoints=8,
    notes="smoke",
   )
  )
 return rows


def build_stage1():
 rows = [
  base_row("s1_sysrisk_samegrid", "stage1_idealized", "idealized_profile", "sysrisk", 5, 2, 0.5, M_SYS, P_FINE, notes="same-grid reference for sampled"),
  base_row("s1_sysrisk_fine", "stage1_idealized", "idealized_profile", "sysrisk", 5, 5, 0.5, M_SYS, P_FINE, notes="fine idealized baseline"),
  base_row("s1_sis_main", "stage1_idealized", "idealized_profile", "sis", 12, 10, 0.5, M_EPI, P_FINE),
  base_row("s1_seir_main", "stage1_idealized", "idealized_profile", "seir", 8, 10, 0.9, M_EPI, P_FINE),
  base_row("s1_lambda_sysrisk", "stage1_idealized", "lambda_sensitivity", "sysrisk", 5, 2, 0.5, [0.3, 0.6], P_FINE),
  base_row("s1_lambda_sis", "stage1_idealized", "lambda_sensitivity", "sis", 12, 10, 0.5, [0.05, 0.1, 0.3], P_FINE),
  base_row("s1_lambda_seir", "stage1_idealized", "lambda_sensitivity", "seir", 8, 10, 0.9, [0.1, 0.3], P_FINE),
 ]
 return rows


def build_stage2():
 rows = []
 for w_lr in (0.75, 0.8, 0.85):
  for updates in (5_000_000, 10_000_000):
   for seed in range(5):
    run_id = f"s2_sysrisk_w{int(w_lr*100):03d}_u{updates//1_000_000:02d}m_seed{seed:02d}"
    rows.append(
     base_row(
      run_id,
      "stage2_sysrisk_screen",
      "sampled_profile",
      "sysrisk",
      5,
      2,
      0.5,
      [0.0, 0.3, 0.6, 1.0],
      P_COARSE,
      num_updates=updates,
      seed=seed,
      w_lr=w_lr,
     )
    )
 return rows


def build_stage3(systemic_w, systemic_updates):
 rows = []
 final_specs = [
  ("sysrisk", 5, 2, 0.5, M_SYS, P_FINE, systemic_updates, systemic_w),
  ("sis", 12, 10, 0.5, M_EPI, P_FINE, 1_000_000, 0.7),
  ("seir", 8, 10, 0.9, M_EPI, P_FINE, 3_000_000, 0.7),
 ]
 for example, n_disc, a_disc, discount, m_grid, p_grid, updates, w_lr in final_specs:
  for seed in range(20):
   run_id = f"s3_{example}_seed{seed:02d}"
   rows.append(
    base_row(
     run_id,
     "stage3_final",
     "sampled_profile",
     example,
     n_disc,
     a_disc,
     discount,
     m_grid,
     p_grid,
     num_updates=updates,
     seed=seed,
     w_lr=w_lr,
     notes="systemic defaults can be replaced after stage2 screen" if example == "sysrisk" else "",
    )
   )
 return rows


def build_stage3_seir_escalation():
 return [
  base_row(
   f"s3b_seir_5m_seed{seed:02d}",
   "stage3_seir_escalation",
   "sampled_profile",
   "seir",
   8,
   10,
   0.9,
   M_EPI,
   P_FINE,
   num_updates=5_000_000,
   seed=seed,
   w_lr=0.7,
   notes="launch only if SEIR stage3 max gap exceeds 0.01",
  )
  for seed in range(10)
 ]


def build_stage4(systemic_w, systemic_updates):
 rows = []
 main_specs = [
  ("sysrisk", 5, 2, 0.5, [0.0, 0.3, 0.6], systemic_updates, systemic_w),
  ("sis", 12, 10, 0.5, [0.0, 0.05, 0.3], 1_000_000, 0.7),
  ("seir", 8, 10, 0.9, [0.0, 0.1, 0.3], 3_000_000, 0.7),
 ]
 theorem_specs = [
  ("sysrisk", 5, 2, 0.3, [0.3], 1_000_000, 0.7),
  ("sis", 12, 10, 0.3, [0.05], 500_000, 0.7),
  ("seir", 8, 10, 0.3, [0.1], 1_000_000, 0.7),
 ]
 for study, specs in (("main", main_specs), ("theorem", theorem_specs)):
  for example, n_disc, a_disc, discount, m_values, updates, w_lr in specs:
   for m_val in m_values:
    for seed in range(10):
     run_id = f"s4_{study}_{example}_m{m_val:g}_seed{seed:02d}".replace(".", "p")
     rows.append(
      base_row(
       run_id,
       "stage4_convergence",
       "convergence",
       example,
       n_disc,
       a_disc,
       discount,
       [m_val],
       [0.0, 0.5, 1.0],
       num_updates=updates,
       seed=seed,
       w_lr=w_lr,
       notes=study,
      )
     )
 return rows


def build_stage5(systemic_w):
 rows = []
 ideal_specs = [
  ("sysrisk", 5, 2, 0.5, M_SYS, P_FINE),
  ("sysrisk", 5, 3, 0.5, M_SYS, P_FINE),
  ("sysrisk", 5, 5, 0.5, M_SYS, P_FINE),
  ("sis", 12, 10, 0.5, M_EPI, P_FINE),
  ("sis", 16, 10, 0.5, M_EPI, P_FINE),
  ("sis", 20, 10, 0.5, M_EPI, P_FINE),
  ("sis", 12, 20, 0.5, M_EPI, P_FINE),
  ("seir", 8, 10, 0.9, M_EPI, P_FINE),
  ("seir", 10, 10, 0.9, M_EPI, P_FINE),
  ("seir", 12, 10, 0.9, M_EPI, P_FINE),
  ("seir", 8, 20, 0.9, M_EPI, P_FINE),
 ]
 for example, n_disc, a_disc, discount, m_grid, p_grid in ideal_specs:
  rows.append(
   base_row(
    f"s5_ideal_{example}_n{n_disc}_a{a_disc}",
    "stage5_grid_sensitivity",
    "idealized_profile",
    example,
    n_disc,
    a_disc,
    discount,
    m_grid,
    p_grid,
   )
  )
 sampled_specs = [
  ("sysrisk", 5, 3, 0.5, [0.0, 0.3, 0.6, 1.0], 10_000_000, systemic_w),
  ("sis", 20, 10, 0.5, [0.0, 0.05, 0.3, 1.0], 1_000_000, 0.7),
  ("seir", 10, 10, 0.9, [0.0, 0.1, 0.3, 1.0], 3_000_000, 0.7),
 ]
 for example, n_disc, a_disc, discount, m_grid, updates, w_lr in sampled_specs:
  for seed in range(5):
   rows.append(
    base_row(
     f"s5_sampled_{example}_n{n_disc}_a{a_disc}_seed{seed:02d}",
     "stage5_grid_sensitivity",
     "sampled_profile",
     example,
     n_disc,
     a_disc,
     discount,
     m_grid,
     P_COARSE,
     num_updates=updates,
     seed=seed,
     w_lr=w_lr,
    )
   )
 return rows


def main():
 parser = argparse.ArgumentParser()
 parser.add_argument("--output-dir", type=str, default=str(CAMPAIGN_DIR / "manifests"))
 parser.add_argument("--systemic-w", type=float, default=0.8)
 parser.add_argument("--systemic-updates", type=int, default=10_000_000)
 args = parser.parse_args()

 out = Path(args.output_dir)
 out.mkdir(parents=True, exist_ok=True)
 stages = {
  "stage0_smoke.csv": build_stage0(),
  "stage1_idealized.csv": build_stage1(),
  "stage2_sysrisk_screen.csv": build_stage2(),
  "stage3_final.csv": build_stage3(args.systemic_w, args.systemic_updates),
  "stage3_seir_escalation.csv": build_stage3_seir_escalation(),
  "stage4_convergence.csv": build_stage4(args.systemic_w, args.systemic_updates),
  "stage5_grid_sensitivity.csv": build_stage5(args.systemic_w),
 }
 all_rows = []
 for name, rows in stages.items():
  write_stage(out / name, rows)
  all_rows.extend(rows)
  print(f"wrote {len(rows):4d} rows -> {out / name}")
 write_stage(out / "all_stages.csv", all_rows)
 print(f"wrote {len(all_rows):4d} rows -> {out / 'all_stages.csv'}")


if __name__ == "__main__":
 main()
