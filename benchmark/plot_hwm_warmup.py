"""benchmark-hwm-chees Fig 3: warmup robustness, peaches vs NUTS.

ESS/grad vs the gradient-sampler warmup length, on ill-conditioned Gaussian
(d=100, kappa=e^12) and Rosenbrock (d=8). peaches re-estimates its preconditioner
from the live ensemble every step, so it needs no mass-matrix warmup and is flat;
NUTS-dense must fit a dense mass during warmup and is far less efficient until
warmup >= 500. 3 sampling seeds on a fixed target; median + min--max band.
"""
import json, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = Path("benchmark/figs"); FIGS.mkdir(parents=True, exist_ok=True)
WARMS = [50, 100, 200, 500, 1000]
SS = [("peaches", "#c0392b", "o", "-", "peaches (ChEES h-walk)"),
      ("nuts", "#2c3e50", "v", "--", "NUTS-dense")]
PANELS = [("gaussian_d100", r"Ill-conditioned Gaussian ($d{=}100$, $\kappa=e^12$)"),
          ("rosenbrock_d8", r"Rosenbrock ($d{=}8$)")]

def seeds(tgt, samp, w):
    v = [json.load(open(f)).get("ess_per_grad")
         for f in glob.glob(f"benchmark/results/{tgt}/{samp}__exp3_w{w}/seed*/metrics.json")]
    return [x for x in v if x is not None and np.isfinite(x)]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, (tgt, title) in zip(axes, PANELS):
    for samp, c, mk, ls, lab in SS:
        ws, med, lo, hi = [], [], [], []
        for w in WARMS:
            v = seeds(tgt, samp, w)
            if v:
                ws.append(w); med.append(np.median(v)); lo.append(min(v)); hi.append(max(v))
        if ws:
            ax.plot(ws, med, marker=mk, color=c, ls=ls, lw=2, ms=8, label=lab, zorder=3)
            ax.fill_between(ws, lo, hi, color=c, alpha=0.18, lw=0, zorder=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(WARMS); ax.set_xticklabels(WARMS)
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("warmup iterations"); ax.set_ylabel("ESS / gradient")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=9, loc="lower right")
    ax.set_title(title, fontsize=11)
fig.suptitle("Warmup robustness: peaches is flat (no mass-matrix warmup); "
             "NUTS-dense needs a long warmup", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIGS / "hwm_warmup.png", dpi=150, bbox_inches="tight")
print("wrote", FIGS / "hwm_warmup.png")
