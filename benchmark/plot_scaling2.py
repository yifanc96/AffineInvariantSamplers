"""Publishable dimensional-scaling figure from the CLEAN `scale3` sweep.

Isotropic Gaussian N(0,I_d), constant walkers/dim = 8d (floor 64), 5 seeds,
scored with the fast ensemble-mean ESS. At a fixed 8 walkers/dim the ChEES
trajectory-length estimate is equally well-resolved at every d, so mean_L grows
smoothly as d^{1/4} and ESS/eval is a clean monotone d^{-1/4} decline with no
low-d hump or floor-transition step. Panel (a) ESS / target evaluation vs d:
peaches ~ d^{-1/4}, emcee ~ d^{-1}. Panel (b) mechanism: ChEES mean trajectory
length. Median over converged seeds (R-hat<=1.03); band = seed min--max.
"""
import json
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = Path("benchmark/figs")
FIGS.mkdir(parents=True, exist_ok=True)
RH = 1.03
TAIL = 8           # d>=TAIL is the pure-8d regime (the 64 floor binds only for d<8)

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 16,
    "legend.fontsize": 12.5, "xtick.labelsize": 13.5, "ytick.labelsize": 13.5,
    "lines.linewidth": 3.0, "lines.markersize": 10,
    "axes.linewidth": 1.4, "xtick.major.width": 1.4, "ytick.major.width": 1.4,
    "figure.dpi": 150, "savefig.dpi": 200,
})

RED, DARK = "#c0392b", "#2c3e50"


def read(samp, metric="ess_per_grad"):
    out = defaultdict(list)
    for f in glob.glob(f"benchmark/results/gaussian_d*/{samp}__scale3/seed*/metrics.json"):
        r = json.load(open(f))
        v = r.get(metric)
        if v is None or not np.isfinite(v):
            continue
        rh = r.get("rhat_max")
        out[r["dim"]].append((v, rh if (rh is not None and np.isfinite(rh)) else None))
    ds = sorted(out)
    y, lo, hi, conv = [], [], [], []
    for d in ds:
        pairs = out[d]
        cv = [v for v, rh in pairs if rh is None or rh <= RH]
        use = cv if cv else [v for v, _ in pairs]
        y.append(np.median(use)); lo.append(min(use)); hi.append(max(use))
        nconv = sum(1 for v, rh in pairs if rh is not None and rh <= RH)
        conv.append(nconv >= (len(pairs) + 1) // 2)
    return (np.array(ds, float), np.array(y), np.array(lo), np.array(hi),
            np.array(conv, bool))


def draw(ax, d, y, lo, hi, conv, c, mk, lab, ls="-"):
    ax.plot(d, y, color=c, lw=3.0, ls=ls, zorder=3, label=lab)
    ax.fill_between(d, lo, hi, color=c, alpha=0.18, lw=0, zorder=1)
    ax.scatter(d[conv], y[conv], color=c, marker=mk, s=95, zorder=4,
               edgecolors="white", linewidths=1.1)
    if (~conv).any():
        ax.scatter(d[~conv], y[~conv], facecolors="white", edgecolors="gray",
                   marker=mk, s=95, zorder=4, linewidths=1.6)


def fit_int(d, y, conv, slope, dmin):
    m = (d >= dmin) & conv
    if m.sum() == 0:
        m = d >= dmin
    return np.mean(np.log(y[m]) - slope * np.log(d[m]))


def free_slope(d, y, conv, dmin):
    m = (d >= dmin) & conv
    return np.polyfit(np.log(d[m]), np.log(y[m]), 1)[0] if m.sum() >= 2 else float("nan")


fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0))

# ---- (a) efficiency ----
dp, yp, lop, hip, cp = read("peaches")
draw(ax[0], dp, yp, lop, hip, cp, RED, "o", "peaches (ChEES h-walk)")
de, ye, loe, hie, ce = read("stretch")
draw(ax[0], de, ye, loe, hie, ce, DARK, "v", "emcee (stretch)")

dd = np.array([TAIL, dp.max()])
ax[0].plot(dd, np.exp(fit_int(dp, yp, cp, -0.25, TAIL)) * dd ** -0.25,
           "k--", lw=1.8, alpha=0.7, zorder=2, label=r"optimal HMC $d^{-1/4}$")
ee = np.array([de.min(), de.max()])
ax[0].plot(ee, np.exp(fit_int(de, ye, ce, -1.0, 8)) * ee ** -1.0,
           "k:", lw=1.8, alpha=0.7, zorder=2, label=r"stretch $d^{-1}$")

sp = free_slope(dp, yp, cp, TAIL)
se = free_slope(de, ye, ce, 8)
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("dimension $d$")
ax[0].set_ylabel("ESS / target evaluation")
ax[0].grid(alpha=0.3, which="both")
ax[0].legend(loc="lower left", framealpha=0.92)
ax[0].set_title("(a) sampling efficiency")
ax[0].text(0.965, 0.50,
           f"peaches slope ${sp:+.2f}$  ($d\\geq{TAIL}$)\nemcee slope ${se:+.2f}$",
           transform=ax[0].transAxes, ha="right", va="center", fontsize=11.5,
           color="#333", bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))

# ---- (b) mechanism: mean_L ----
dL, yL, loL, hiL, cL = read("peaches", "mean_L")
ax[1].plot(dL, yL, color=RED, lw=3.0, marker="o", ms=10, zorder=3,
           label="peaches ChEES length", markeredgecolor="white", markeredgewidth=1.1)
ax[1].fill_between(dL, loL, hiL, color=RED, alpha=0.18, lw=0, zorder=1)
ddl = np.array([TAIL, dL.max()])
ax[1].plot(ddl, np.exp(fit_int(dL, yL, cL, 0.25, TAIL)) * ddl ** 0.25,
           "k--", lw=1.8, alpha=0.7, zorder=2, label=r"$d^{+1/4}$")
sL = free_slope(dL, yL, cL, TAIL)
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("dimension $d$")
ax[1].set_ylabel(r"mean trajectory length $\bar L$")
ax[1].grid(alpha=0.3, which="both")
ax[1].legend(loc="upper left", framealpha=0.92)
ax[1].set_title("(b) ChEES trajectory length")
ax[1].text(0.97, 0.03, f"slope ${sL:+.2f}$  ($d\\geq{TAIL}$)",
           transform=ax[1].transAxes, ha="right", va="bottom", fontsize=11.5,
           color="#333", bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))

fig.tight_layout()
fig.savefig(FIGS / "pn_scaling.png", bbox_inches="tight")
print("wrote", FIGS / "pn_scaling.png")
print(f"peaches epg: {list(zip(dp.astype(int), np.round(yp,4)))}")
print(f"emcee   epg: {list(zip(de.astype(int), np.round(ye,5)))}")
print(f"peaches L  : {list(zip(dL.astype(int), np.round(yL,2)))}")
print(f"slopes: peaches epg {sp:+.3f} (d>={TAIL}), emcee {se:+.3f}, L {sL:+.3f}")
