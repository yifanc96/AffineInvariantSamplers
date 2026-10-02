"""peaches-note Fig 3: warmup robustness -- peaches vs NUTS-dense (no peanuts).

ESS/grad vs the gradient-sampler warmup length, on ill-conditioned Gaussian
(d=100, kappa=e^6, 400 walkers, tag exp3) and Rosenbrock (d=8, 256 walkers, tag
exp3b). 8 sampling seeds. Line = median over CONVERGED seeds (R-hat<=1.03); band =
interquartile range (25-75%) over those seeds -- IQR rather than min--max because
ChEES's integration length is bimodal on the banana. Marker hollow if a minority
of a curve's seeds converged.
"""
import json, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = Path("benchmark/figs"); FIGS.mkdir(parents=True, exist_ok=True)
WARMS = [50, 100, 200, 500, 1000]
RH = 1.03
YLAB = "ESS / target evaluation"

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 16,
    "legend.fontsize": 12.5, "xtick.labelsize": 13.5, "ytick.labelsize": 13.5,
    "lines.linewidth": 3.0, "lines.markersize": 10,
    "axes.linewidth": 1.4, "xtick.major.width": 1.4, "ytick.major.width": 1.4,
    "figure.dpi": 150, "savefig.dpi": 200,
})
SS = [("peaches", "#c0392b", "o", "-", "peaches (ChEES h-walk)"),
      ("nuts", "#2c3e50", "v", "--", "NUTS-dense")]
PANELS = [("gaussian_d100", r"Ill-conditioned Gaussian ($d{=}100$, $\kappa=e^6$)", "warmf"),
          ("rosenbrock_d8", r"Rosenbrock ($d{=}8$)", "warmfb")]


def agg(tgt, samp, tag, w):
    pairs = []
    for f in glob.glob(f"benchmark/results/{tgt}/{samp}__{tag}_w{w}/seed*/metrics.json"):
        r = json.load(open(f)); eg = r.get("ess_per_grad"); rh = r.get("rhat_max")
        if eg is None or not np.isfinite(eg):
            continue
        pairs.append((eg, rh if (rh is not None and np.isfinite(rh)) else None))
    if not pairs:
        return None
    cv = [v for v, rh in pairs if rh is None or rh <= RH]
    use = cv if cv else [v for v, _ in pairs]
    nconv = sum(1 for v, rh in pairs if rh is not None and rh <= RH)
    return (np.median(use), np.percentile(use, 25), np.percentile(use, 75),
            nconv >= (len(pairs) + 1) // 2)


fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
for ax, (tgt, title, tag) in zip(axes, PANELS):
    for samp, c, mk, ls, lab in SS:
        xs, med, lo, hi, conv = [], [], [], [], []
        for w in WARMS:
            a = agg(tgt, samp, tag, w)
            if a:
                xs.append(w); med.append(a[0]); lo.append(a[1]); hi.append(a[2]); conv.append(a[3])
        if not xs:
            continue
        xs = np.array(xs, float); med = np.array(med); conv = np.array(conv)
        ax.plot(xs, med, color=c, ls=ls, lw=3.0, zorder=3, label=lab)
        ax.fill_between(xs, lo, hi, color=c, alpha=0.16, lw=0, zorder=1)
        cm = conv.astype(bool)
        ax.scatter(xs[cm], med[cm], color=c, marker=mk, s=95, zorder=4,
                   edgecolors="white", linewidths=1.1)
        if (~cm).any():
            ax.scatter(xs[~cm], med[~cm], facecolors="white", edgecolors="gray", marker=mk,
                       s=95, zorder=4, linewidths=1.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(WARMS); ax.set_xticklabels(WARMS)
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("warmup iterations"); ax.set_ylabel(YLAB)
    ax.grid(alpha=0.3, which="both"); ax.legend(loc="lower right", framealpha=0.92)
    ax.set_title(title)
fig.suptitle("Warmup robustness: peaches needs no mass-matrix warmup",
             fontsize=17, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIGS / "pn_warmup.png", dpi=150, bbox_inches="tight")
print("wrote", FIGS / "pn_warmup.png")
