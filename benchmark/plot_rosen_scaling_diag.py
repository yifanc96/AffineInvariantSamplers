"""Diagnostic: why the paired Rosenbrock does NOT show a clean d^{-1/4} over the
accessible range, even though it is formally a product of identical banana pairs.

(a) ESS/gradient vs d. The curve does not track d^{-1/4} (shown for reference):
    at n_walkers=4d the low-d runs are walker-starved (d=2: 8 walkers on a single
    extreme banana -> near-failure), so efficiency RISES with d as walkers
    accumulate, swamping the dimensional law. Wide seed bands (the target is hard).
(b) The mechanism: peaches' ChEES-adapted mean trajectory length is large and
    erratic on the curved banana (10-160) versus the smooth, slowly-growing length
    on the Gaussian (~6-9). The varying curvature has no single good integration
    length, so per-gradient efficiency is both low and noisy.

3 seeds, median + min--max band; hollow = R-hat>1.03. Rosenbrock peaches from
scaleLb/scaleL (uniform 4d); Gaussian peaches for contrast; peanuts from exp1.
"""
import json, glob
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = Path("benchmark/figs"); FIGS.mkdir(parents=True, exist_ok=True)
RH_CONV = 1.03


def read_runs(pattern, tag, metric="ess_per_grad"):
    """dim -> list of (value, rhat_or_None), one entry per seed."""
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
    """Median/band over CONVERGED seeds only; point marked converged iff a
    majority of its seeds converged (R-hat<=1.03). Stuck seeds never count."""
    ds = sorted(m)
    y, lo, hi, conv = [], [], [], []
    for d in ds:
        pairs = m[d]
        cv = [v for (v, rh) in pairs if rh is None or rh <= RH_CONV]
        use = cv if cv else [v for (v, _) in pairs]
        y.append(np.median(use)); lo.append(min(use)); hi.append(max(use))
        nconv = sum(1 for (v, rh) in pairs if rh is not None and rh <= RH_CONV)
        conv.append(nconv >= (len(pairs) + 1) // 2)
    return np.array(ds, float), np.array(y), np.array(lo), np.array(hi), np.array(conv)


def draw(ax, d, y, lo, hi, conv, c, mk, lab, ls="-"):
    ax.plot(d, y, color=c, lw=2, ls=ls, zorder=3, label=lab)
    ax.fill_between(d, lo, hi, color=c, alpha=0.15, lw=0, zorder=1)
    cm = conv.astype(bool)
    ax.scatter(d[cm], y[cm], color=c, marker=mk, s=58, zorder=4)
    if (~cm).any():
        ax.scatter(d[~cm], y[~cm], facecolors="white", edgecolors="gray", marker=mk,
                   s=58, zorder=4, linewidths=1.3)


def rosen(samp, metric, *tags):
    srcs = [read_runs(f"benchmark/results/rosenbrock_d*/{samp}__{t}/seed*/metrics.json", t, metric) for t in tags]
    return to_arrays(merge(*srcs))


def gauss(samp, metric):
    # 4d uniform (chosen protocol): scaleLb low-d + scaleL d>=32
    b = read_runs(f"benchmark/results/gaussian_d*/{samp}__scaleLb/seed*/metrics.json", "scaleLb", metric)
    sL = read_runs(f"benchmark/results/gaussian_d*/{samp}__scaleL/seed*/metrics.json", "scaleL", metric)
    e4 = read_runs(f"benchmark/results/gaussian_d*/{samp}__exp4c/seed*/metrics.json", "exp4c", metric)
    return to_arrays(merge(b, sL, e4))


fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))

# ---- (a) Rosenbrock ESS/grad -- does not track d^{-1/4} ---------------------
rp = rosen("peaches", "ess_per_grad", "scaleLd", "scaleLb", "scaleL")
rn = rosen("peanuts", "ess_per_grad", "exp1")
draw(ax[0], *rp, "#c0392b", "o", "peaches")
draw(ax[0], *rn, "#8e44ad", "s", "peanuts (exp1)")
d, y, _, _, conv = rp
dd = np.array([d.min(), d.max()])
# d^{-1/4} anchored at the high-d end, shown only as the asymptotic EXPECTATION
ax[0].plot(dd, y[-1] * (dd / d[-1]) ** -0.25, "k--", lw=1.3, alpha=0.5,
           label=r"$d^{-1/4}$ (asymptotic expectation)")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("dimension $d$"); ax[0].set_ylabel("ESS / gradient")
ax[0].grid(alpha=0.3, which="both"); ax[0].legend(fontsize=8.5, loc="lower right")
ax[0].set_title("(a) Rosenbrock efficiency does not track $d^{-1/4}$ at accessible $d$")

# ---- (b) mechanism: erratic mean_L on the banana vs smooth on the Gaussian --
draw(ax[1], *rosen("peaches", "mean_L", "scaleLd", "scaleLb", "scaleL"), "#c0392b", "o", "Rosenbrock (curved)")
draw(ax[1], *gauss("peaches", "mean_L"), "#2980b9", "o", "Gaussian (smooth)", ls="--")
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("dimension $d$"); ax[1].set_ylabel(r"peaches mean trajectory length $\bar L$")
ax[1].grid(alpha=0.3, which="both"); ax[1].legend(fontsize=8.5, loc="upper left")
ax[1].set_title(r"(b) ChEES length is erratic on curvature, smooth on the Gaussian")

fig.suptitle("Paired Rosenbrock: a product of bananas, but per-block curvature "
             "(not dimension) sets the cost, masking the $d^{-1/4}$ law", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIGS / "pp_rosen_scaling_diag.png", dpi=150, bbox_inches="tight")
print("wrote", FIGS / "pp_rosen_scaling_diag.png")
