"""d^{-1/4} dimensional-scaling figure for the isotropic-Gaussian sweep (exp4).

For optimally-tuned HMC / kinetic samplers (Beskos et al. 2013) the
acceptance-optimal step size scales as eps ~ d^{-1/4}, hence the trajectory
length L ~ d^{1/4} and the per-gradient efficiency ESS/grad ~ d^{-1/4}.

Produces figs/exp4_scaling.png with three log-log panels (ESS/grad, mean_L,
step size vs d), each with the theoretical reference slope and the
least-squares fitted exponent per sampler.
"""
from __future__ import annotations
import json, glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("benchmark/results")
FIGS = Path("benchmark/figs"); FIGS.mkdir(parents=True, exist_ok=True)
SAMPLERS = [("peaches", "#c0392b", "o"), ("peanuts", "#8e44ad", "v"),
            ("nuts", "#27ae60", "D")]


def load_exp4():
    rows = [json.loads(Path(f).read_text())
            for f in glob.glob(str(ROOT / "**/metrics.json"), recursive=True)]
    return [r for r in rows if r.get("tag") == "exp4"]


def median_by_d(rows, sampler, metric):
    g = defaultdict(list)
    for r in rows:
        if r["sampler"] != sampler:
            continue
        v = r.get(metric)
        if v is not None and np.isfinite(v):
            g[r["dim"]].append(v)
    ds = sorted(g)
    return np.array(ds, float), np.array([np.median(g[d]) for d in ds])


def fit_slope(d, y):
    """Least-squares slope of log y vs log d."""
    m = (d > 0) & (y > 0)
    if m.sum() < 2:
        return float("nan")
    return np.polyfit(np.log(d[m]), np.log(y[m]), 1)[0]


def main():
    rows = load_exp4()
    if not rows:
        print("no exp4 rows"); return
    print(f"loaded {len(rows)} exp4 cells")

    panels = [("ess_per_grad", -0.25, "ESS / gradient", "d^{-1/4}"),
              ("mean_L",        0.25, "mean trajectory length  L̄", "d^{1/4}"),
              ("final_step_size", -0.25, "adapted step size  ε", "d^{-1/4}")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    summary = {}
    for ax, (metric, ref_slope, ylab, reflab) in zip(axes, panels):
        for s, c, mk in SAMPLERS:
            d, y = median_by_d(rows, s, metric)
            if len(d) == 0:
                continue
            slope = fit_slope(d, y)
            summary[(metric, s)] = slope
            ax.plot(d, y, marker=mk, color=c, lw=1.8, ms=7,
                    label=f"{s} (fit {slope:+.2f})")
        # theoretical reference line through the peaches anchor
        dref = np.array([rows_d for rows_d in sorted({r['dim'] for r in rows})], float)
        dp, yp = median_by_d(rows, "peaches", metric)
        if len(dp):
            anchor = yp[0] * (dref / dp[0]) ** ref_slope
            ax.plot(dref, anchor, "k--", lw=1.2, alpha=0.7,
                    label=f"theory {reflab}")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("dimension d"); ax.set_ylabel(ylab)
        ax.set_title(metric); ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle("Dimensional scaling on isotropic N(0,I_d): "
                 "eps~d^{-1/4}, L~d^{1/4}, ESS/grad~d^{-1/4} "
                 "(Beskos et al. 2013)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGS / "exp4_scaling.png", dpi=140, bbox_inches="tight")
    print("wrote", FIGS / "exp4_scaling.png")
    print("\nFitted log-log slopes (theory: ess/grad -0.25, mean_L +0.25, step -0.25):")
    for (metric, s), slope in sorted(summary.items()):
        print(f"  {metric:<16s} {s:<9s} {slope:+.3f}")


if __name__ == "__main__":
    main()
