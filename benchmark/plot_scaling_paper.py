"""Paper scaling figure for peaches & peanuts (corrected dual averaging, exp4c).

Simple and direct: show the as-run curve (d >= DMIN, asymptotic regime) against
the theoretical d^{-1/4} (resp. d^{+1/4}) slope, anchored to the same starting
point. The data tracks the theory slope -- no fitted line needed.
  (L) adapted step size  eps  ~ d^{-1/4}
  (M) mean trajectory    Lbar ~ d^{+1/4}
  (R) per-gradient eff.   ESS/grad ~ d^{-1/4}
"""
import json, glob
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = Path("benchmark/figs"); FIGS.mkdir(parents=True, exist_ok=True)
SS = [("peaches", "#c0392b", "o"), ("peanuts", "#8e44ad", "v")]
TAG = "exp4c"
DMIN = 32   # show only the asymptotic regime

rows = [json.loads(Path(f).read_text())
        for f in glob.glob("benchmark/results/**/metrics.json", recursive=True)]
rows = [r for r in rows if r.get("tag") == TAG]

def by_d(s, m):
    g = defaultdict(list)
    for r in rows:
        v = r.get(m)
        if r["sampler"] == s and v is not None and np.isfinite(v) and r["dim"] >= DMIN:
            g[r["dim"]].append(v)
    ds = sorted(g)
    return np.array(ds, float), np.array([np.median(g[d]) for d in ds])

def Lbar(s):
    d, L = by_d(s, "mean_L")
    if len(d) == 0:
        d, L = by_d(s, "mean_num_steps")
    return d, L

def panel(ax, getter, ref_exp, title, ylab):
    for s, c, mk in SS:
        d, y = getter(s)
        if len(d) < 2:
            continue
        ax.plot(d, y, marker=mk, color=c, lw=2, ms=8, label=s, zorder=3)
        # theory slope anchored to this curve's first point
        ax.plot(d, y[0] * (d / d[0]) ** ref_exp, "--", color=c, lw=1.3,
                alpha=0.55, zorder=2)
    ax.plot([], [], "k--", lw=1.3, alpha=0.7, label=f"theory $d^{{{ref_exp:+.2f}}}$")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("dimension $d$"); ax.set_ylabel(ylab)
    ax.set_title(title); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=9)

fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(15, 4.4))
panel(axL, lambda s: by_d(s, "final_step_size"), -0.25,
      "Step size", r"adapted $\epsilon$")
panel(axM, Lbar, +0.25, "Trajectory length", r"mean leapfrog $\bar L$")
panel(axR, lambda s: by_d(s, "ess_per_grad"), -0.25,
      "Per-gradient efficiency", "ESS / gradient")
fig.suptitle("Dimensional scaling on isotropic Gaussian (corrected dual averaging): "
             "curves track the theoretical $d^{\\mp1/4}$ slope", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIGS / "scaling_paper.png", dpi=150, bbox_inches="tight")
print("wrote", FIGS / "scaling_paper.png")
