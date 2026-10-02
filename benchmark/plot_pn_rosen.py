"""peaches-note Fig 2: dimensional scaling on the paired Rosenbrock (no peanuts).

The paired Rosenbrock is formally a product of d/2 identical banana pairs, so in
principle it obeys the same d^{-1/4} law -- but it does not over accessible d.
(a) peaches ESS/gradient stays ~1e-3 and does NOT fall as d^{-1/4} (shown for
reference); emcee fails to converge (hollow). (b) the mechanism: peaches' adapted
trajectory length is large and erratic on the curved banana (the bimodal ChEES
length) versus smooth and slowly-growing on the Gaussian.

peaches from the tuned converged runs (256 walkers d>=8 / 512 d=2,4, tag scaleLd);
emcee from exp1. Median/band over CONVERGED seeds only; hollow = minority
converged.
"""
import json, glob
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = Path("benchmark/figs"); FIGS.mkdir(parents=True, exist_ok=True)
RH = 1.03
YLAB = "ESS / target evaluation"

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 16,
    "legend.fontsize": 12, "xtick.labelsize": 13.5, "ytick.labelsize": 13.5,
    "lines.linewidth": 3.0, "lines.markersize": 10,
    "axes.linewidth": 1.4, "xtick.major.width": 1.4, "ytick.major.width": 1.4,
    "figure.dpi": 150, "savefig.dpi": 200,
})


def read_runs(pattern, tag, metric="ess_per_grad"):
    out = defaultdict(list)
    for f in glob.glob(pattern):
        r = json.load(open(f))
        if tag is not None and r.get("tag") != tag:
            continue
        v = r.get(metric)
        if v is None or not np.isfinite(v) or v <= 0:
            continue
        rh = r.get("rhat_max")
        out[r["dim"]].append((v, rh if (rh is not None and np.isfinite(rh)) else None))
    return out


def merge(*sources):
    m = {}
    for src in sources:
        for d, rec in src.items():
            if d not in m and rec:
                m[d] = rec
    return m


def to_arrays(m):
    ds = sorted(m)
    y, lo, hi, conv = [], [], [], []
    for d in ds:
        pairs = m[d]
        cv = [v for v, rh in pairs if rh is None or rh <= RH]
        use = cv if cv else [v for v, _ in pairs]
        y.append(np.median(use)); lo.append(min(use)); hi.append(max(use))
        nconv = sum(1 for v, rh in pairs if rh is not None and rh <= RH)
        conv.append(nconv >= (len(pairs) + 1) // 2)
    return np.array(ds, float), np.array(y), np.array(lo), np.array(hi), np.array(conv)


def draw(ax, d, y, lo, hi, conv, c, mk, lab, ls="-"):
    ax.plot(d, y, color=c, lw=3.0, ls=ls, zorder=3, label=lab)
    ax.fill_between(d, lo, hi, color=c, alpha=0.15, lw=0, zorder=1)
    cm = conv.astype(bool)
    ax.scatter(d[cm], y[cm], color=c, marker=mk, s=95, zorder=4,
               edgecolors="white", linewidths=1.1)
    if (~cm).any():
        ax.scatter(d[~cm], y[~cm], facecolors="white", edgecolors="gray", marker=mk,
                   s=95, zorder=4, linewidths=1.6)


def rosen(samp, metric, *tags):
    srcs = [read_runs(f"benchmark/results/rosenbrock_d*/{samp}__{t}/seed*/metrics.json", t, metric) for t in tags]
    return to_arrays(merge(*srcs))


def gauss(samp, metric):
    s2 = read_runs(f"benchmark/results/gaussian_d*/{samp}__scale3/seed*/metrics.json", "scale3", metric)
    return to_arrays(s2)


fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0))

# (a) Rosenbrock efficiency vs d -- does not track d^{-1/4}
rp = rosen("peaches", "ess_per_grad", "rz8")
re = rosen("stretch", "ess_per_grad", "rz8")
draw(ax[0], *rp, "#c0392b", "o", "peaches (ChEES h-walk)")
draw(ax[0], *re, "#27ae60", "D", "emcee (stretch)")
d, y = rp[0], rp[1]
dd = np.array([d.min(), d.max()])
ax[0].plot(dd, y[-1] * (dd / d[-1]) ** -0.25, "k--", lw=1.8, alpha=0.6,
           label=r"$d^{-1/4}$ (asymptotic expectation)")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("dimension $d$"); ax[0].set_ylabel(YLAB)
ax[0].grid(alpha=0.3, which="both"); ax[0].legend(loc="lower right", framealpha=0.92)
ax[0].set_title(r"(a) efficiency does not track $d^{-1/4}$")

# (b) mechanism: erratic ChEES length on the banana vs smooth on the Gaussian
draw(ax[1], *rosen("peaches", "mean_L", "rz8"), "#c0392b", "o", "Rosenbrock (curved)")
draw(ax[1], *gauss("peaches", "mean_L"), "#2980b9", "o", "Gaussian (smooth)", ls="--")
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("dimension $d$"); ax[1].set_ylabel(r"mean trajectory length $\bar L$")
ax[1].grid(alpha=0.3, which="both"); ax[1].legend(loc="upper left", framealpha=0.92)
ax[1].set_title(r"(b) ChEES length: erratic on curvature")

fig.suptitle("Paired Rosenbrock: per-block curvature (not dimension) sets the cost",
             fontsize=17, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIGS / "pn_rosen.png", dpi=150, bbox_inches="tight")
print("wrote", FIGS / "pn_rosen.png")
