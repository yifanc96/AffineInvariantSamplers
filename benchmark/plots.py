"""Aggregate benchmark results -> tables (CSV) + figures.

Reads benchmark/results/**/metrics.json, computes 3-seed medians, and emits:
  figs/exp1_dim_scaling.png   (IAT + bias vs D, peaches vs stretch)
  figs/exp2_affine.png        (IAT vs kappa, peaches vs chess vs stretch)
  figs/exp3_warmup.png        (bias + IAT vs warmup, peaches vs NUTS)
  results/summary.csv         (all cells, 3-seed median)
"""
from __future__ import annotations

import json
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("benchmark/results")
FIGS = Path("benchmark/figs")
FIGS.mkdir(parents=True, exist_ok=True)


def load_all():
    rows = []
    for f in glob.glob(str(ROOT / "**/metrics.json"), recursive=True):
        rows.append(json.loads(Path(f).read_text()))
    return rows


def median_over_seeds(rows, key_fields, metric):
    """Median of `metric` grouped by key_fields tuple."""
    groups = defaultdict(list)
    for r in rows:
        if metric not in r or r[metric] is None:
            continue
        k = tuple(r[f] for f in key_fields)
        groups[k].append(r[metric])
    return {k: float(np.median(v)) for k, v in groups.items()}


# ───────────────────────── Exp 1: dimensional scaling ──────────────────────

def plot_exp1(rows):
    r1 = [r for r in rows if r["tag"] == "exp1"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for col, target in enumerate(("gaussian", "rosenbrock")):
        sub = [r for r in r1 if r["target"] == target]
        for metric, ax in (("iat_max", axes[0, col]),
                           ("mean_err_sigma_max", axes[1, col])):
            med = median_over_seeds(sub, ("sampler", "dim"), metric)
            for s, c, mk in (("peaches", "#c0392b", "o"), ("stretch", "#2980b9", "s")):
                pts = sorted((d, v) for (ss, d), v in med.items() if ss == s)
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, marker=mk, color=c, label=s, lw=2, ms=7)
            ax.set_yscale("log"); ax.set_xscale("log" if target == "gaussian" else "linear")
            ax.set_xlabel("dimension D")
            ax.set_ylabel({"iat_max": "max IAT (slowest dim)",
                           "mean_err_sigma_max": "max mean error [σ]"}[metric])
            ax.set_title(f"{target}: {'mixing' if metric=='iat_max' else 'bias'}")
            ax.grid(alpha=0.3, which="both"); ax.legend()
    fig.suptitle("Exp 1 — dimensional scaling: peaches vs stretch "
                 "(lower is better; 3-seed median)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIGS / "exp1_dim_scaling.png", dpi=130, bbox_inches="tight")
    print("wrote", FIGS / "exp1_dim_scaling.png")


# ───────────────────────── Exp 2: affine invariance ────────────────────────

def plot_exp2(rows):
    r2 = [r for r in rows if str(r["tag"]).startswith("exp2")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for metric, ax in (("iat_max", axes[0]), ("ess_min", axes[1])):
        med = median_over_seeds(r2, ("sampler", "log_cond"), metric)
        for s, c, mk in (("peaches", "#c0392b", "o"), ("chess", "#16a085", "^"),
                         ("stretch", "#2980b9", "s")):
            pts = sorted((np.exp(2 * lc), v) for (ss, lc), v in med.items() if ss == s)
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker=mk, color=c, label=s, lw=2, ms=7)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("covariance condition number κ")
        ax.set_ylabel({"iat_max": "max IAT", "ess_min": "min ESS"}[metric])
        ax.set_title("mixing vs conditioning" if metric == "iat_max"
                     else "efficiency vs conditioning")
        ax.grid(alpha=0.3, which="both"); ax.legend()
    fig.suptitle("Exp 2 — affine invariance (Gaussian D=50): peaches/stretch "
                 "κ-invariant, chess degrades", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGS / "exp2_affine.png", dpi=130, bbox_inches="tight")
    print("wrote", FIGS / "exp2_affine.png")


# ───────────────────────── Exp 3: warmup robustness ────────────────────────

def _warm_from_tag(tag):
    return int(str(tag).split("_w")[-1])


def plot_exp3(rows):
    r3 = [r for r in rows if str(r["tag"]).startswith("exp3_w")]
    for r in r3:
        r["_warm"] = _warm_from_tag(r["tag"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    # Top row: ESS per gradient (work-normalized; raw IAT unfairly credits
    # NUTS for its many-leapfrog-per-sample tree expansion). Higher = better.
    # Bottom row: mean bias.
    for col, target in enumerate(("gaussian", "rosenbrock")):
        sub = [r for r in r3 if r["target"] == target]
        for metric, ax in (("ess_per_grad", axes[0, col]),
                           ("mean_err_sigma_max", axes[1, col])):
            med = median_over_seeds(sub, ("sampler", "_warm"), metric)
            for s, c, mk in (("peaches", "#c0392b", "o"), ("peanuts", "#8e44ad", "v"),
                             ("nuts", "#27ae60", "D")):
                pts = sorted((w, v) for (ss, w), v in med.items() if ss == s)
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, marker=mk, color=c, label=s, lw=2, ms=7)
            ax.set_yscale("log"); ax.set_xscale("log")
            ax.set_xlabel("warmup iterations")
            ax.set_ylabel({"ess_per_grad": "ESS / gradient (higher better)",
                           "mean_err_sigma_max": "max mean error [σ]"}[metric])
            ax.set_title(f"{target}: {'efficiency' if metric=='ess_per_grad' else 'bias'}")
            ax.grid(alpha=0.3, which="both"); ax.legend()
    fig.suptitle("Exp 3 — warmup robustness: NUTS efficiency collapses at short "
                 "warmup (frozen mass matrix); peaches keeps adapting", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIGS / "exp3_warmup.png", dpi=130, bbox_inches="tight")
    print("wrote", FIGS / "exp3_warmup.png")


def write_summary_csv(rows):
    import csv
    fields = ["target", "dim", "sampler", "tag", "log_cond", "warmup",
              "acceptance_rate", "mean_L", "mean_err_sigma_max",
              "std_rel_err_max", "iat_max", "iat_median", "ess_min",
              "ess_per_grad", "wall_s", "seed"]
    out = ROOT / "summary.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["target"], r["dim"],
                                             r["sampler"], str(r["tag"]), r["seed"])):
            w.writerow(r)
    print("wrote", out, f"({len(rows)} rows)")


def main():
    rows = load_all()
    print(f"loaded {len(rows)} metrics.json")
    if not rows:
        return
    write_summary_csv(rows)
    plot_exp1(rows)
    plot_exp2(rows)
    plot_exp3(rows)


if __name__ == "__main__":
    main()
