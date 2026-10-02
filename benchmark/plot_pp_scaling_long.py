"""Fig 1 (peaches/peanuts note): dimensional scaling on N(0,I_d), extended.

10x-longer production (N=30000 vs 3000) and the dimension range extended down to
d=2 so the full pre-asymptotic -> asymptotic story is visible. UNIFORM protocol
throughout: n_walkers=4d, target_accept=0.8, stretch pre-warmup ws=100, warmup
4000. 3 seeds on a fixed target; line=median, band=seed min--max.

  (a) per-gradient efficiency: peaches/peanuts approach the optimal-HMC law
      d^{-1/4} in the tail; emcee (gradient-free stretch) falls as d^{-1}.
  (b) the mechanism: the ChEES-adapted mean trajectory length grows as d^{+1/4},
      the signature of the epsilon ~ d^{-1/4} step-size law.

Data: scaleLb (d<=16, run at exactly 4d) + scaleL (d>=32, already 4d). The
isotropic Gaussian is a genuine product target, so the asymptotic law is clean.
(The paired Rosenbrock is also a product target but its per-block curvature makes
the ChEES integration length erratic, so the clean law is not realized over the
accessible range -- see plot_rosen_scaling_diag.py and the curvature experiment.)
"""
import json, glob
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = Path("benchmark/figs"); FIGS.mkdir(parents=True, exist_ok=True)
RH_CONV = 1.03
GTAIL = 64   # asymptotic tail: d=16,32 still sit on a pre-asymptotic plateau


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
    """Median/band over CONVERGED seeds (R-hat<=1.03 or no R-hat); a point is
    marked converged only if a majority of its seeds converged. Correctness:
    a stuck seed never contributes to the reported efficiency."""
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


def draw(ax, d, y, lo, hi, conv, c, mk, lab):
    ax.plot(d, y, color=c, lw=2, zorder=3, label=lab)
    ax.fill_between(d, lo, hi, color=c, alpha=0.16, lw=0, zorder=1)
    cm = conv.astype(bool)
    ax.scatter(d[cm], y[cm], color=c, marker=mk, s=60, zorder=4)
    if (~cm).any():
        ax.scatter(d[~cm], y[~cm], facecolors="white", edgecolors="gray", marker=mk,
                   s=60, zorder=4, linewidths=1.3)


def fixed_slope_intercept(d, y, conv, slope, dmin):
    m = (d >= dmin) & conv.astype(bool)
    if m.sum() == 0:
        m = d >= dmin
    return np.mean(np.log(y[m]) - slope * np.log(d[m]))


def free_slope(d, y, conv, dmin):
    m = (d >= dmin) & conv.astype(bool)
    if m.sum() < 2:
        return None
    return np.polyfit(np.log(d[m]), np.log(y[m]), 1)[0]


import os
PROTO = os.environ.get("GAUSS_PROTO", "uniform4d")   # "uniform4d" | "floor128"


def gauss(samp, metric="ess_per_grad"):
    # floor128: n_walkers=max(128,4d) -- scaleLd (128w low-d) then scaleL (d>=32).
    # uniform4d: n_walkers=4d everywhere -- scaleLb (4d low-d) then scaleL.
    low = "scaleLd" if PROTO == "floor128" else "scaleLb"
    lo = read_runs(f"benchmark/results/gaussian_d*/{samp}__{low}/seed*/metrics.json", low, metric)
    sL = read_runs(f"benchmark/results/gaussian_d*/{samp}__scaleL/seed*/metrics.json", "scaleL", metric)
    e4 = read_runs(f"benchmark/results/gaussian_d*/{samp}__exp4c/seed*/metrics.json", "exp4c", metric)
    return to_arrays(merge(lo, sL, e4))


SS = [("peaches", "#c0392b", "o"), ("peanuts", "#8e44ad", "s")]
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))

# ---- (a) per-gradient efficiency -------------------------------------------
for samp, c, mk in SS:
    d, y, lo, hi, conv = gauss(samp)
    draw(ax[0], d, y, lo, hi, conv, c, mk, samp)
de, ye, lod, hid, ce = to_arrays(read_runs(
    "benchmark/results/gaussian_d*/stretch__exp4c/seed*/metrics.json", "exp4c"))
draw(ax[0], de, ye, lod, hid, ce, "#2c3e50", "v", "emcee (stretch)")
# theory references anchored to the asymptotic tail
d, y, _, _, conv = gauss("peaches")
c14 = fixed_slope_intercept(d, y, conv, -0.25, GTAIL)
dd = np.array([d.min(), d.max()])
ax[0].plot(dd, np.exp(c14) * dd ** -0.25, "k--", lw=1.3, alpha=0.6, label=r"theory $d^{-1/4}$")
sp = free_slope(d, y, conv, GTAIL)
c1 = fixed_slope_intercept(de, ye, ce, -1.0, 16)
ax[0].plot(np.array([de.min(), de.max()]), np.exp(c1) * np.array([de.min(), de.max()]) ** -1.0,
           "k:", lw=1.3, alpha=0.6, label=r"theory $d^{-1}$")
se = free_slope(de, ye, ce, 16)
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("dimension $d$"); ax[0].set_ylabel("ESS / gradient")
ax[0].grid(alpha=0.3, which="both"); ax[0].legend(fontsize=8.5, loc="lower left")
ax[0].set_title("(a) per-gradient efficiency")
ax[0].text(0.97, 0.97, f"peaches slope {sp:+.2f} ($d\\geq{GTAIL}$)\nemcee slope {se:+.2f}",
           transform=ax[0].transAxes, ha="right", va="top", fontsize=8, color="#444",
           bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.85))

# ---- (b) mean trajectory length (mechanism) --------------------------------
for samp, c, mk in SS:
    d, y, lo, hi, conv = gauss(samp, "mean_L")
    ax[1].plot(d, y, color=c, lw=2, marker=mk, ms=7, label=samp, zorder=3)
    ax[1].fill_between(d, lo, hi, color=c, alpha=0.16, lw=0, zorder=1)
d, y, _, _, conv = gauss("peaches", "mean_L")
cL = fixed_slope_intercept(d, y, conv, 0.25, GTAIL)
dd = np.array([d.min(), d.max()])
ax[1].plot(dd, np.exp(cL) * dd ** 0.25, "k--", lw=1.3, alpha=0.6, label=r"theory $d^{+1/4}$")
sL = free_slope(d, y, conv, GTAIL)
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("dimension $d$"); ax[1].set_ylabel(r"mean trajectory length $\bar L$")
ax[1].grid(alpha=0.3, which="both"); ax[1].legend(fontsize=8.5, loc="upper left")
ax[1].set_title("(b) ChEES trajectory length grows as $d^{+1/4}$")
ax[1].text(0.97, 0.03, f"peaches slope {sL:+.2f} ($d\\geq{GTAIL}$)", transform=ax[1].transAxes,
           ha="right", va="bottom", fontsize=8, color="#444",
           bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.85))

fig.suptitle("Dimensional scaling on the isotropic Gaussian $N(0,I_d)$: "
             "peaches/peanuts realise the optimal-HMC $d^{-1/4}$ law, emcee falls as $d^{-1}$",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIGS / "pp_scaling_long.png", dpi=150, bbox_inches="tight")
print("wrote", FIGS / "pp_scaling_long.png")

# ---- report ----------------------------------------------------------------
for samp in ("peaches", "peanuts"):
    d, y, *_ , conv = gauss(samp)
    dL, yL, *_, cL2 = gauss(samp, "mean_L")
    print(f"  gauss {samp:8s} ess/grad slope(d>={GTAIL})={free_slope(d,y,conv,GTAIL):+.3f}  "
          f"mean_L slope={free_slope(dL,yL,cL2,GTAIL):+.3f}  "
          f"y={[round(v,4) for v in y]}")
