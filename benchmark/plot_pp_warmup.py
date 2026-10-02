"""Fig 3 (peaches/peanuts note): warmup robustness.

peaches/peanuts vs NUTS-dense, ESS/grad vs the gradient-sampler warmup length, on
ill-conditioned Gaussian (d=100, kappa=e^6, rotated) and Rosenbrock (d=8).

8 sampling seeds on a fixed target. Gaussian: 400 walkers (tag exp3). Rosenbrock:
256 walkers (tag exp3b) -- more walkers than the 128 default for cleaner
short-warmup convergence. Line = median over CONVERGED seeds (R-hat<=1.03); band =
interquartile range (25-75%) over those seeds. We use the IQR, not min--max,
because ChEES's integration length is bimodal on the banana (some seeds find the
long trajectory, others a short one) -- both converge, but a min--max band would
span an order of magnitude and bury the warmup-flatness message. Marker hollow if
a minority of a curve's seeds converged.
"""
import json, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = Path("benchmark/figs"); FIGS.mkdir(parents=True, exist_ok=True)
WARMS = [50, 100, 200, 500, 1000]
RH = 1.03
SS = [("peaches", "#c0392b", "o", "-"), ("peanuts", "#8e44ad", "s", "-"),
      ("nuts", "#2c3e50", "v", "--")]
# (target_dir, title, tag)
PANELS = [("gaussian_d100", r"Ill-conditioned Gaussian ($d{=}100$, $\kappa=e^6$)", "exp3"),
          ("rosenbrock_d8", r"Rosenbrock ($d{=}8$)", "exp3b")]


def agg(tgt, samp, tag, w):
    """(median, 25th pct, 75th pct, majority-converged) over converged seeds."""
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


fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, (tgt, title, tag) in zip(axes, PANELS):
    for samp, c, mk, ls in SS:
        xs, med, lo, hi, conv = [], [], [], [], []
        for w in WARMS:
            a = agg(tgt, samp, tag, w)
            if a:
                xs.append(w); med.append(a[0]); lo.append(a[1]); hi.append(a[2]); conv.append(a[3])
        if not xs:
            continue
        lab = {"peaches": "peaches", "peanuts": "peanuts", "nuts": "NUTS-dense"}[samp]
        xs = np.array(xs, float); med = np.array(med); conv = np.array(conv)
        ax.plot(xs, med, color=c, ls=ls, lw=2, zorder=3, label=lab)
        ax.fill_between(xs, lo, hi, color=c, alpha=0.16, lw=0, zorder=1)
        cm = conv.astype(bool)
        ax.scatter(xs[cm], med[cm], color=c, marker=mk, s=58, zorder=4)
        if (~cm).any():
            ax.scatter(xs[~cm], med[~cm], facecolors="white", edgecolors="gray",
                       marker=mk, s=58, zorder=4, linewidths=1.3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(WARMS); ax.set_xticklabels(WARMS)
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("warmup iterations"); ax.set_ylabel("ESS / gradient")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=9, loc="lower right")
    ax.set_title(title, fontsize=11)
fig.suptitle("Warmup robustness: peaches/peanuts are flat; NUTS-dense needs a long warmup "
             "to fit its mass matrix", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIGS / "pp_warmup.png", dpi=150, bbox_inches="tight")
print("wrote", FIGS / "pp_warmup.png")
