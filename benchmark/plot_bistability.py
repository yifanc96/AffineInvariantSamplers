"""ChEES trajectory-length bistability on the banana (Rosenbrock).

Three panels that together make the case:
 (a) the adapted mean length is BIMODAL on the d=2 banana (clean gap: a short
     "stuck" mode and a long "ridge-traversing" mode) -- many seeds pooled;
 (b) the long mode is better on BOTH axes at once -- lower error vs truth AND
     higher ESS/eval -- so it is not a speed/accuracy trade-off; points colored
     by whether rank-split R-hat would pass (<=1.03), showing R-hat does NOT
     separate the modes;
 (c) the long-mode fraction is essentially flat in warmup length and in init
     scale -- so the bistability is intrinsic, not cured by longer tuning.

Reads the d=2 study (tags b2_w*_i*) and the d=8 warmup/walker scans (rwarm_*,
rwscan_*). Written to benchmark/figs/pn_bistability.png.
"""
import json, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = Path("benchmark/figs"); FIGS.mkdir(parents=True, exist_ok=True)
LGAP = 60          # L threshold separating short (stuck) from long (traverse)
RH = 1.03

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14.5,
    "legend.fontsize": 11.5, "xtick.labelsize": 12.5, "ytick.labelsize": 12.5,
    "lines.linewidth": 2.6, "lines.markersize": 9,
    "axes.linewidth": 1.3, "xtick.major.width": 1.3, "ytick.major.width": 1.3,
    "figure.dpi": 150, "savefig.dpi": 200,
})
RED, BLUE, GRAY = "#c0392b", "#2980b9", "#7f8c8d"


def load(patterns):
    rows = []
    for pat in patterns:
        for f in glob.glob(pat):
            r = json.load(open(f))
            if r.get("mean_L") is None or r.get("ess_per_grad") is None:
                continue
            rows.append(r)
    return rows


# ---- data ----
d2 = load(["benchmark/results/rosenbrock_d2/peaches__b2_*/seed*/metrics.json"])
# d=2 fixed config for the clean histogram (warmup 8000, init 2)
d2_hist = load(["benchmark/results/rosenbrock_d2/peaches__b2_w8000_i2/seed*/metrics.json"])
scatter = load(["benchmark/results/rosenbrock_d2/peaches__b2_*/seed*/metrics.json",
                "benchmark/results/rosenbrock_d8/peaches__rwarm_*/seed*/metrics.json",
                "benchmark/results/rosenbrock_d8/peaches__rwscan_*/seed*/metrics.json"])

fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))

# (a) L histogram -- bimodality
L2 = [r["mean_L"] for r in (d2_hist or d2)]
ax[0].hist(L2, bins=np.logspace(np.log10(2), np.log10(600), 26), color=RED, alpha=0.8,
           edgecolor="white")
ax[0].axvline(LGAP, color="k", ls=":", lw=1.6)
ax[0].set_xscale("log")
ax[0].set_xlabel(r"adapted mean trajectory length $\bar L$")
ax[0].set_ylabel("number of runs")
ax[0].set_title(r"(a) $\bar L$ is bimodal ($d{=}2$ banana)")
ax[0].text(0.06, 0.9, "short\n(stuck)", transform=ax[0].transAxes, ha="left", va="top",
           fontsize=11, color=GRAY)
ax[0].text(0.94, 0.9, "long\n(traverse)", transform=ax[0].transAxes, ha="right", va="top",
           fontsize=11, color=RED)

# (b) accuracy AND efficiency vs L -- long is better on both; color = R-hat pass
L = np.array([r["mean_L"] for r in scatter])
err = np.array([r.get("std_rel_err_max", np.nan) for r in scatter])
eps = np.array([r["ess_per_grad"] for r in scatter])
rh = np.array([r.get("rhat_max", np.nan) for r in scatter])
passed = rh <= RH
sc = ax[1].scatter(L, err, c=eps, cmap="viridis", s=70, edgecolors="k", linewidths=0.5,
                   norm=plt.matplotlib.colors.LogNorm(), zorder=3)
ax[1].scatter(L[~passed], err[~passed], facecolors="none", edgecolors="red", s=150,
              linewidths=1.8, zorder=4, label=r"$\hat R>1.03$")
ax[1].axvline(LGAP, color="k", ls=":", lw=1.6)
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel(r"mean trajectory length $\bar L$")
ax[1].set_ylabel(r"std-error vs truth (worst coord)")
ax[1].set_title("(b) long mode: accurate and efficient")
cb = fig.colorbar(sc, ax=ax[1]); cb.set_label("ESS / eval", fontsize=11)
ax[1].legend(loc="upper right", framealpha=0.9)

# (c) per-seed adapted L vs warmup (d=8) -- more warmup does NOT converge the mode
warms = [4000, 8000, 16000, 32000]
for sd in range(6):
    xs, ys = [], []
    for w in warms:
        try:
            r = json.load(open(f"benchmark/results/rosenbrock_d8/peaches__rwarm_w{w}/seed{sd}/metrics.json"))
            if r.get("mean_L") is not None:
                xs.append(w); ys.append(r["mean_L"])
        except FileNotFoundError:
            pass
    if xs:
        ys = np.array(ys)
        ax[2].plot(xs, ys, "-o", lw=2.0, ms=8, alpha=0.85,
                   color=(RED if ys[-1] > LGAP else BLUE))
ax[2].axhspan(2, LGAP, color=BLUE, alpha=0.07); ax[2].axhspan(LGAP, 400, color=RED, alpha=0.07)
ax[2].axhline(LGAP, color="k", ls=":", lw=1.4)
ax[2].set_xscale("log"); ax[2].set_yscale("log")
ax[2].set_xticks(warms); ax[2].set_xticklabels(warms)
ax[2].xaxis.set_minor_formatter(plt.NullFormatter())
ax[2].set_xlabel(r"warmup iterations ($d{=}8$, one line per seed)")
ax[2].set_ylabel(r"adapted mean length $\bar L$")
ax[2].text(0.5, 0.93, "long (traverse)", transform=ax[2].transAxes, ha="center",
           color=RED, fontsize=11)
ax[2].text(0.5, 0.06, "short (stuck)", transform=ax[2].transAxes, ha="center",
           color=BLUE, fontsize=11)
ax[2].set_title("(c) more warmup does not fix it")

fig.suptitle("ChEES trajectory-length bistability on the curved banana: a stochastic, "
             "intrinsic failure mode that $\\hat R$ does not flag",
             fontsize=16, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIGS / "pn_bistability.png", bbox_inches="tight")
print("wrote", FIGS / "pn_bistability.png")
print(f"d2 hist n={len(L2)}")
# mode accuracy/efficiency summary
for lab, msk in [("LONG", L > LGAP), ("short", L <= LGAP)]:
    if msk.any():
        print(f"{lab}: n={msk.sum()} med std_err={np.nanmedian(err[msk]):.4f} med ess/eval={np.nanmedian(eps[msk]):.5f} rhat_pass={np.mean(rh[msk]<=RH):.2f}")
