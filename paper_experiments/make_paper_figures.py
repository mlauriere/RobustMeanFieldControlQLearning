#!/usr/bin/env python3
"""Regenerate paper figures from included aggregate data.

Reads aggregates from ../paper_results/ and writes figures to ../paper_figures/.

Output:
  Regenerated main figures (PNG + PDF):
    main_profiles_{sysrisk,sis,seir}.{png,pdf}
    convergence_q_error.{png,pdf}

  These are the robustness profiles and Q-function convergence figures
  used in the paper's numerical section.  Figure conventions match the
  paper: reward (negated cost), robustness radius m, eval. drift zeta,
  dashed idealized references, sampled uncertainty bands, no global
  titles.

  Relative-improvement plots are also regenerated as supplementary
  diagnostics:
    relative_improvement_{sysrisk,sis,seir}.{png,pdf}

Prebuilt supplementary figures (convergence diagnostics, landscapes,
grid sensitivity, etc.) in paper_figures/ are not overwritten by this
script and remain as included HPC-generated assets.
"""
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "paper_results"
FIGURES_DIR = SCRIPT_DIR.parent / "paper_figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR.parent))

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def read_csv(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def bootstrap_ci(values, rng_seed=123, n_boot=2000, alpha=0.05):
    """Bootstrap mean and (alpha/2, 1-alpha/2) CI."""
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


def color_for_p(p_val):
    return plt.cm.viridis(0.08 + 0.84 * float(p_val))


ENV_ORDER = ["sysrisk", "sis", "seir"]
ENV_LABELS = {"sysrisk": "Systemic Risk", "sis": "SIS", "seir": "SEIR"}

CONV_COLORS = {
    0.0: "#1b9e77", 0.05: "#d95f02", 0.1: "#7570b3",
    0.3: "#e7298a", 0.5: "#a6761d", 1.0: "#1f78b4",
}


def convergence_color(m_val):
    return CONV_COLORS.get(
        round(float(m_val), 3),
        plt.cm.tab10(abs(hash(round(float(m_val), 3))) % 10),
    )


def grid_key(r):
    return (int(float(r.get("n_disc", 0))), int(float(r.get("a_disc", 0))),
            float(r.get("discount", 0)))


# -------------------------------------------------------------------
# Main paper figure: robustness profiles
# -------------------------------------------------------------------

def plot_main_profiles():
    """Robustness profiles with dashed idealized references and uncertainty bands.

    Uses final_policy_costs.csv for sampled curves and same-grid idealized
    references.  Plots expected discounted reward (negated cost).
    """
    path = RESULTS_DIR / "final_policy_costs.csv"
    if not path.exists():
        print("  SKIP main_profiles: final_policy_costs.csv not found")
        return

    all_rows = read_csv(path)

    # Separate sampled and idealized rows
    sampled = [r for r in all_rows if r.get("solver", "") == "sampled"]
    ideals = [r for r in all_rows if r.get("solver", "") == "idealized"]

    # Also load sampled_vs_idealized_gaps.csv as fallback for ideal curves
    gap_rows = []
    gap_path = RESULTS_DIR / "sampled_vs_idealized_gaps.csv"
    if gap_path.exists():
        gap_rows = read_csv(gap_path)

    for example in ENV_ORDER:
        ex_sampled = [r for r in sampled if r["example"] == example]
        if not ex_sampled:
            continue

        # Determine sampled grid size for matching idealized
        sampled_grids = sorted({grid_key(r) for r in ex_sampled})
        pref_grid = sampled_grids[0]

        fig, ax = plt.subplots(figsize=(8.8, 5.8))
        p_values = sorted({float(r["p"]) for r in ex_sampled})

        for p_val in p_values:
            # --- sampled curve with bootstrap CI ---
            ms = sorted({float(r["M"]) for r in ex_sampled
                         if abs(float(r["p"]) - p_val) < 1e-12})
            means, lows, highs = [], [], []
            for m_val in ms:
                vals = [-float(r["cost"]) for r in ex_sampled
                        if abs(float(r["p"]) - p_val) < 1e-12
                        and abs(float(r["M"]) - m_val) < 1e-12]
                mean, lo, hi = bootstrap_ci(vals)
                means.append(mean)
                lows.append(lo)
                highs.append(hi)
            color = color_for_p(p_val)
            ax.plot(ms, means, color=color, marker="o", markersize=5,
                    linewidth=2.0, label=rf"$\zeta={p_val:g}$")
            ax.fill_between(ms, lows, highs, color=color, alpha=0.13,
                            linewidth=0)

            # --- dashed idealized reference (same grid) ---
            ex_ideal = [r for r in ideals if r["example"] == example
                        and abs(float(r["p"]) - p_val) < 1e-12]
            # Prefer same-grid idealized rows
            same_grid = [r for r in ex_ideal if grid_key(r) == pref_grid]
            if same_grid:
                ex_ideal = same_grid
            if ex_ideal:
                ideal_by_m = defaultdict(list)
                for r in ex_ideal:
                    ideal_by_m[float(r["M"])].append(-float(r["cost"]))
                ideal_m = sorted(ideal_by_m)
                ideal_r = [np.mean(ideal_by_m[m]) for m in ideal_m]
                ax.plot(ideal_m, ideal_r, color=color, linestyle="--",
                        linewidth=1.5, alpha=0.75)

        ax.set_xlabel("robustness radius m")
        ax.set_ylabel("expected discounted reward")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(title=r"eval. drift $\zeta$",
                  bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=True)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(FIGURES_DIR / f"main_profiles_{example}.{ext}",
                        dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"  main_profiles_{example}.png  main_profiles_{example}.pdf")


# -------------------------------------------------------------------
# Main paper figure: Q-function convergence
# -------------------------------------------------------------------

def plot_convergence():
    """Q-function convergence (log-log) from convergence_summary.csv.

    Uses only rows with study == \"main\" (the publication campaign).
    Fails with a clear error if no main-study rows are present, to
    prevent smoke-test data from silently producing a paper figure.
    """
    path = RESULTS_DIR / "convergence_summary.csv"
    if not path.exists():
        print("  SKIP convergence: convergence_summary.csv not found")
        return False

    all_rows = read_csv(path)
    main_rows = [r for r in all_rows if r.get("study", "") == "main"]
    if not main_rows:
        print("  ERROR: convergence_summary.csv has no rows with study='main'.")
        print("  The main convergence aggregate is required for the paper figure.")
        print("  Available studies:", sorted({r.get("study","") for r in all_rows}))
        return False

    n_used = 0
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=False)
    for ax, example in zip(axes, ENV_ORDER):
        rows_ex = [r for r in main_rows if r["example"] == example]
        if not rows_ex:
            ax.set_axis_off()
            continue
        n_used += len(rows_ex)
        guide_drawn = False
        for m_val in sorted({float(r["M"]) for r in rows_ex}):
            rows_m = [r for r in rows_ex
                      if abs(float(r["M"]) - m_val) < 1e-12]
            grouped = defaultdict(list)
            for r in rows_m:
                grouped[float(r["step"])].append(float(r["q_sup_error"]))
            steps = np.array(sorted(grouped), dtype=float)
            if steps.size == 0:
                continue
            med = np.array([np.median(grouped[s]) for s in steps],
                           dtype=float)
            lo = np.array([np.quantile(grouped[s], 0.1) for s in steps],
                          dtype=float)
            hi = np.array([np.quantile(grouped[s], 0.9) for s in steps],
                          dtype=float)
            med = np.maximum(med, 1e-12)
            lo = np.maximum(lo, 1e-12)
            hi = np.maximum(hi, 1e-12)
            color = convergence_color(m_val)
            ax.plot(steps, med, color=color, linewidth=2.0,
                    label=rf"$m={m_val:g}$")
            ax.fill_between(steps, lo, hi, color=color, alpha=0.15,
                            linewidth=0)
            if not guide_drawn and steps.size > 1:
                w_lr = float(rows_m[0].get("w_lr", 0.7) or 0.7)
                guide = np.sqrt(np.log(np.maximum(steps, 3.0))
                                / np.maximum(steps, 1.0) ** w_lr)
                guide = guide / guide[0] * med[0]
                ax.plot(steps, guide, color="black", linestyle="--",
                        linewidth=1.3, alpha=0.65,
                        label=rf"$\sqrt{{\log T/T^{{{w_lr:.2g}}}}}$")
                guide_drawn = True
        ax.set_title(ENV_LABELS.get(example, example))
        ax.set_xlabel("sampled updates")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.legend(frameon=True, fontsize=8)
    axes[0].set_ylabel(r"$\|\check Q_T-\check Q^*_m\|_\infty$")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"convergence_q_error.{ext}",
                    dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  convergence_q_error.png  convergence_q_error.pdf  ({n_used} main-study rows)")
    return True


# -------------------------------------------------------------------
# Supplementary: relative improvement
# -------------------------------------------------------------------

def plot_relative_improvement():
    """Relative improvement over non-robust baseline (supplementary)."""
    path = RESULTS_DIR / "relative_improvements.csv"
    if not path.exists():
        print("  SKIP relative_improvement: file not found")
        return

    rows = read_csv(path)
    for example in ENV_ORDER:
        subset = [r for r in rows if r["example"] == example]
        if not subset:
            continue
        fig, ax = plt.subplots(figsize=(8.4, 5.4))
        for p_val in sorted({float(r["p"]) for r in subset}):
            rows_p = [r for r in subset
                      if abs(float(r["p"]) - p_val) < 1e-12]
            rows_p.sort(key=lambda r: float(r["M"]))
            ax.plot(
                [float(r["M"]) for r in rows_p],
                [100.0 * float(r["relative_improvement_mean"])
                 for r in rows_p],
                marker="o", color=color_for_p(p_val),
                linewidth=2.0, label=rf"$\zeta={p_val:g}$",
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("robustness radius m")
        ax.set_ylabel("relative improvement (%)")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(title=r"eval. drift $\zeta$",
                  bbox_to_anchor=(1.02, 1.0), loc="upper left",
                  frameon=True)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(
                FIGURES_DIR / f"relative_improvement_{example}.{ext}",
                dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"  relative_improvement_{example}.png"
              f"  relative_improvement_{example}.pdf")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    n_figs = 0

    # --- summary of available data ---
    print("Regenerating paper figures from aggregate data ...")
    print()

    # profile data summary
    prof_path = RESULTS_DIR / "final_policy_costs.csv"
    if prof_path.exists():
        prof_rows = read_csv(prof_path)
        for solver in ["sampled", "idealized"]:
            for example in ENV_ORDER:
                sub = [r for r in prof_rows
                       if r.get("solver") == solver and r["example"] == example]
                if sub:
                    n_seeds = len({r.get("seed") for r in sub}) - (1 if "" in {r.get("seed") for r in sub} else 0)
                    n_M = len({float(r["M"]) for r in sub})
                    n_p = len({float(r["p"]) for r in sub})
                    print(f"  profile data: {example:8s} {solver:10s}"
                          f"  rows={len(sub):5d}  seeds={n_seeds:2d}"
                          f"  M={n_M}  p={n_p}")

    # convergence data summary
    conv_path = RESULTS_DIR / "convergence_summary.csv"
    if conv_path.exists():
        conv_rows = read_csv(conv_path)
        main = [r for r in conv_rows if r.get("study") == "main"]
        for example in ENV_ORDER:
            sub = [r for r in main if r["example"] == example]
            if sub:
                steps = sorted({float(r["step"]) for r in sub})
                Ms = sorted({float(r["M"]) for r in sub})
                seeds = sorted({r["seed"] for r in sub})
                print(f"  conv. data:   {example:8s} main        "
                      f"  rows={len(sub):5d}  seeds={len(seeds):2d}"
                      f"  steps={steps[0]:.0f}..{steps[-1]:.0f}"
                      f"  M={[f'{m:g}' for m in Ms]}")

    print()

    # --- generate figures ---
    print("Main paper figures:")
    plot_main_profiles()
    # Each example produces 2 files (png+pdf): 3 examples → 6 files
    n_figs += 6
    print()

    ok = plot_convergence()
    if ok:
        n_figs += 2  # png + pdf
    print()

    print("Supplementary:")
    plot_relative_improvement()
    n_figs += 6  # 3 examples x (png+pdf)
    print()

    print(f"Done.  {n_figs} files written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
