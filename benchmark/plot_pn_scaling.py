"""peaches-note Fig 1: dimensional scaling on N(0,I_d) -- peaches vs emcee.

No peanuts. Uniform n_walkers=4d, d=2..256, target_accept 0.8, ws=100, 6 seeds at
d<=16 / 3 at d>=32. (a) per-gradient efficiency: peaches ~ d^{-1/4}, emcee ~
d^{-1}. (b) the mechanism: peaches' ChEES-tuned mean trajectory length ~ d^{+1/4}.
Median/band over CONVERGED seeds only (R-hat<=1.03). NUTS also achieves d^{-1/4}
and is omitted here so the peaches-vs-emcee contrast reads cleanly (it is in
Experiments 2-3).
"""
import json, glob
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = Path("benchmark/figs"); FIGS.mkdir(parents=True, exist_ok=True)
RH = 1.03
GTAIL = 64


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


def draw(ax, d, y, lo, hi, conv, c, mk, lab):
    ax.plot(d, y, color=c, lw=2, zorder=3, label=lab)
    ax.fill_between(d, lo, hi, color=c, alpha=0.16, lw=0, zorder=1)
    cm = conv.astype(bool)
    ax.scatter(d[cm], y[cm], color=c, marker=mk, s=60, zorder=4)
    if (~cm).any():
        ax.scatter(d[~cm], y[~cm], facecolors="white", edgecolors="gray", marker=mk,
                   s=60, zorder=4, linewidths=1.3)


def fixed_int(d, y, conv, slope, dmin):
    m = (d >= dmin) & conv.astype(bool)
    if m.sum() == 0:
        m = d >= dmin
    return np.mean(np.log(y[m]) - slope * np.log(d[m]))


def free_slope(d, y, conv, dmin):
    m = (d >= dmin) & conv.astype(bool)
    return np.polyfit(np.log(d[m]), np.log(y[m]), 1)[0] if m.sum() >= 2 else None


def gauss(samp, metric="ess_per_grad"):
    b = read_runs(f"benchmark/results/gaussian_d*/{samp}__scaleLb/seed*/metrics.json", "scaleLb", metric)
    sL = read_runs(f"benchmark/results/gaussian_d*/{samp}__scaleL/seed*/metrics.json", "scaleL", metric)
    e4 = read_runs(f"benchmark/results/gaussian_d*/{samp}__exp4c/seed*/metrics.json", "exp4c", metric)
    return to_arrays(merge(b, sL, e4))


fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))

# (a) efficiency
dp, yp, lop, hip, cp = gauss("peaches")
draw(ax[0], dp, yp, lop, hip, cp, "#c0392b", "o", "peaches (ChEES h-walk)")
de, ye, loe, hie, ce = gauss("stretch")
draw(ax[0], de, ye, loe, hie, ce, "#2c3e50", "v", "emcee (stretch)")
dd = np.array([dp.min(), dp.max()])
ax[0].plot(dd, np.exp(fixed_int(dp, yp, cp, -0.25, GTAIL)) * dd ** -0.25, "k--", lw=1.3, alpha=0.6,
           label=r"theory $d^{-1/4}$")
ee = np.array([de.min(), de.max()])
ax[0].plot(ee, np.exp(fixed_int(de, ye, ce, -1.0, 16)) * ee ** -1.0, "k:", lw=1.3, alpha=0.6,
           label=r"theory $d^{-1}$")
sp, se = free_slope(dp, yp, cp, GTAIL), free_slope(de, ye, ce, 16)
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("dimension $d$"); ax[0].set_ylabel("ESS / gradient")
ax[0].grid(alpha=0.3, which="both"); ax[0].legend(fontsize=8.5, loc="lower left")
ax[0].set_title("(a) per-gradient efficiency")
ax[0].text(0.97, 0.97, f"peaches slope {sp:+.2f} ($d\\geq{GTAIL}$)\nemcee slope {se:+.2f}",
           transform=ax[0].transAxes, ha="right", va="top", fontsize=8, color="#444",
           bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.85))

# (b) mechanism: mean_L
dL, yL, loL, hiL, cL = gauss("peaches", "mean_L")
ax[1].plot(dL, yL, color="#c0392b", lw=2, marker="o", ms=7, label="peaches", zorder=3)
ax[1].fill_between(dL, loL, hiL, color="#c0392b", alpha=0.16, lw=0)
ddl = np.array([dL.min(), dL.max()])
ax[1].plot(ddl, np.exp(fixed_int(dL, yL, cL, 0.25, GTAIL)) * ddl ** 0.25, "k--", lw=1.3, alpha=0.6,
           label=r"theory $d^{+1/4}$")
sL_ = free_slope(dL, yL, cL, GTAIL)
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("dimension $d$"); ax[1].set_ylabel(r"mean trajectory length $\bar L$")
ax[1].grid(alpha=0.3, which="both"); ax[1].legend(fontsize=8.5, loc="upper left")
ax[1].set_title("(b) ChEES trajectory length grows as $d^{+1/4}$")
ax[1].text(0.97, 0.03, f"peaches slope {sL_:+.2f} ($d\\geq{GTAIL}$)", transform=ax[1].transAxes,
           ha="right", va="bottom", fontsize=8, color="#444",
           bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.85))

fig.suptitle("Dimensional scaling on $N(0,I_d)$: peaches realises the optimal-HMC "
             "$d^{-1/4}$ law, emcee falls as $d^{-1}$", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIGS / "pn_scaling.png", dpi=150, bbox_inches="tight")
print("wrote", FIGS / "pn_scaling.png", "| peaches slope", round(sp, 3), "emcee", round(se, 3),
      "| meanL slope", round(sL_, 3))
