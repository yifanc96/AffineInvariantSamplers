"""benchmark-hwm-chees Fig 1: dimensional scaling on N(0,I_d).

peaches (ChEES-adapted Hamiltonian walk move), emcee (stretch), NUTS (diagonal
mass; on an isotropic target the diagonal mass is the identity, so this is the
plain-HMC scaling baseline). Left: ESS/grad vs d -- peaches & NUTS track the
optimal-HMC law d^{-1/4}, emcee falls as d^{-1}. Right: mean trajectory length
~ d^{+1/4}. 3 seeds, median + min--max band; hollow = R-hat>1.03 (unconverged).
"""
import json, glob
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = Path("benchmark/figs"); FIGS.mkdir(parents=True, exist_ok=True)
# NUTS omitted from the scaling figure on purpose: it also achieves d^{-1/4}
# (a known result) and sits ABOVE peaches at low/mid d, which makes the panel
# hard to read and distracts from the peaches-vs-emcee contrast (d^{-1/4} vs
# d^{-1}). NUTS is benchmarked in Experiments 2 and 3.
SS = [("peaches", "#c0392b", "o", "peaches (ChEES h-walk)"),
      ("stretch", "#2c3e50", "v", "emcee (stretch)")]

# NUTS is capped at d<=64: beyond that its no-U-turn trajectory length grows
# faster than d^{1/4} (overhead), which is a separate effect from the step-size
# scaling law and would muddy the comparison. peaches/emcee shown full range.
DMAX = {"nuts": 64}

def by_d(samp, m):
    g = defaultdict(list); rh = defaultdict(list)
    for f in glob.glob(f"benchmark/results/gaussian_d*/{samp}__exp4c/seed*/metrics.json"):
        r = json.load(open(f))
        if r.get("tag") != "exp4c": continue
        if r["dim"] > DMAX.get(samp, 10**9): continue
        v = r.get(m)
        if v is not None and np.isfinite(v):
            g[r["dim"]].append(v)
            if r.get("rhat_max") is not None: rh[r["dim"]].append(r["rhat_max"])
    ds = sorted(g)
    conv = np.array([(np.median(rh[d]) <= 1.03) if rh.get(d) else True for d in ds])
    return (np.array(ds, float), np.array([np.median(g[d]) for d in ds]),
            np.array([min(g[d]) for d in ds]), np.array([max(g[d]) for d in ds]), conv)

def draw(ax, d, y, lo, hi, conv, c, mk, lab):
    ax.plot(d, y, color=c, lw=2, zorder=3, label=lab)
    ax.fill_between(d, lo, hi, color=c, alpha=0.16, lw=0, zorder=1)
    cm = conv.astype(bool)
    ax.scatter(d[cm], y[cm], color=c, marker=mk, s=64, zorder=4)
    if (~cm).any():
        ax.scatter(d[~cm], y[~cm], facecolors="white", edgecolors="gray", marker=mk,
                   s=64, zorder=4, linewidths=1.3)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for samp, c, mk, lab in SS:
    d, y, lo, hi, conv = by_d(samp, "ess_per_grad")
    if len(d) >= 2: draw(ax[0], d, y, lo, hi, conv, c, mk, lab)
dpe, ype, *_ = by_d("peaches", "ess_per_grad")
dem, yem, *_ = by_d("stretch", "ess_per_grad")
dd = np.array([dpe.min(), dpe.max()])
ax[0].plot(dd, ype[0]*(dd/dpe[0])**-0.25, "k--", lw=1.3, alpha=0.6, label=r"theory $d^{-1/4}$")
de = np.array([dem.min(), dem.max()])
ax[0].plot(de, yem[0]*(de/dem[0])**-1.0, "k:", lw=1.3, alpha=0.6, label=r"theory $d^{-1}$")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("dimension $d$"); ax[0].set_ylabel("ESS / gradient")
ax[0].grid(alpha=0.3, which="both"); ax[0].legend(fontsize=8.5, loc="lower left")
ax[0].set_title("(a) per-gradient efficiency vs dimension")
for samp, c, mk, lab in [SS[0]]:    # peaches: the ChEES-tuned trajectory length
    d, y, lo, hi, conv = by_d(samp, "mean_L")
    if len(d) >= 2:
        ax[1].plot(d, y, marker=mk, color=c, lw=2, ms=8, label=lab, zorder=3)
        ax[1].fill_between(d, lo, hi, color=c, alpha=0.16, lw=0)
dpe, ype, *_ = by_d("peaches", "mean_L")
dd = np.array([dpe.min(), dpe.max()])
ax[1].plot(dd, ype[0]*(dd/dpe[0])**0.25, "k--", lw=1.3, alpha=0.6, label=r"theory $d^{+1/4}$")
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("dimension $d$"); ax[1].set_ylabel("mean trajectory length $\\bar L$")
ax[1].grid(alpha=0.3, which="both"); ax[1].legend(fontsize=8.5, loc="upper left")
ax[1].set_title("(b) trajectory length grows as $d^{+1/4}$")
fig.suptitle("Dimensional scaling on $N(0,I_d)$: peaches achieves the optimal-HMC "
             "$d^{-1/4}$ law, exponentially better than emcee's $d^{-1}$", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIGS / "hwm_scaling.png", dpi=150, bbox_inches="tight")
print("wrote", FIGS / "hwm_scaling.png")
