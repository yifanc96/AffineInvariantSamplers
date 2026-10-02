"""benchmark-hwm-chees Fig 2: conditioning vs curvature.

Left: ill-conditioned Gaussian d=50 vs kappa. peaches & emcee are affine-invariant
(kappa-flat for free); NUTS is kappa-flat only WITH mass adaptation -- NUTS with
the mass fixed to the identity (no adaptation) collapses as kappa grows. Right:
Rosenbrock vs d. peaches and NUTS (gradient methods, local dynamics) cross the
curved ridge; emcee (single affine move) fails to converge. On Rosenbrock the
mass adaptation is irrelevant (global conditioning is mild), so we show the
no-mass NUTS as the NUTS representative.

3 seeds; median + min--max band. solid = converged (R-hat<=1.03), hollow = not.
"""
import json, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = Path("benchmark/figs"); FIGS.mkdir(parents=True, exist_ok=True)

def series(specs):
    xs, med, lo, hi, conv = [], [], [], [], []
    for x, tgt, samp, tag in specs:
        eg = [json.load(open(f)).get("ess_per_grad")
              for f in glob.glob(f"benchmark/results/{tgt}/{samp}__{tag}/seed*/metrics.json")]
        rh = [json.load(open(f)).get("rhat_max")
              for f in glob.glob(f"benchmark/results/{tgt}/{samp}__{tag}/seed*/metrics.json")]
        eg = [v for v in eg if v is not None and np.isfinite(v)]
        rh = [v for v in rh if v is not None]
        if not eg: continue
        xs.append(x); med.append(np.median(eg)); lo.append(min(eg)); hi.append(max(eg))
        conv.append((np.median(rh) <= 1.03) if rh else True)
    return (np.array(xs, float), np.array(med), np.array(lo), np.array(hi), np.array(conv))

def plot_s(ax, x, med, lo, hi, conv, c, mk, ls, lab):
    if len(x) == 0: return
    ax.plot(x, med, color=c, ls=ls, lw=2, zorder=3, label=lab)
    ax.fill_between(x, lo, hi, color=c, alpha=0.10, lw=0, zorder=1)
    cm = conv.astype(bool)
    ax.scatter(x[cm], med[cm], color=c, marker=mk, s=58, zorder=4)
    if (~cm).any():
        ax.scatter(x[~cm], med[~cm], facecolors="white", edgecolors="gray", marker=mk,
                   s=58, zorder=4, linewidths=1.3)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
KAP = [2, 4, 6, 8]
for samp, tagf, c, mk, ls, lab in [
        ("peaches", "exp2_lc{}", "#c0392b", "o", "-", "peaches (ChEES h-walk)"),
        ("stretch", "exp2_lc{}", "#27ae60", "D", "-", "emcee (stretch)"),
        ("nuts", "exp2_lc{}", "#2980b9", "^", "-", "NUTS (dense mass)"),
        ("nuts", "idmass_lc{}", "#2c3e50", "v", "--", "NUTS (no mass adapt.)")]:
    x, med, lo, hi, conv = series([(k, "gaussian_d50", samp, tagf.format(k)) for k in KAP])
    plot_s(ax[0], x, med, lo, hi, conv, c, mk, ls, lab)
ax[0].set_yscale("log"); ax[0].set_xticks(KAP)
ax[0].set_xlabel(r"$\log\kappa$ (condition number $=e^{\log\kappa}$)")
ax[0].set_ylabel("ESS / gradient"); ax[0].grid(alpha=0.3, which="both")
ax[0].legend(fontsize=8, loc="lower left")
ax[0].set_title(r"(a) Ill-conditioned Gaussian ($d{=}50$) vs $\kappa$")
DR = [8, 16, 32]
for samp, tag, c, mk, ls, lab in [
        ("peaches", "exp1", "#c0392b", "o", "-", "peaches (ChEES h-walk)"),
        ("stretch", "exp1", "#27ae60", "D", "-", "emcee (stretch)"),
        ("nuts", "idmass", "#2c3e50", "v", "--", "NUTS")]:
    x, med, lo, hi, conv = series([(d, f"rosenbrock_d{d}", samp, tag) for d in DR])
    plot_s(ax[1], x, med, lo, hi, conv, c, mk, ls, lab)
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xticks(DR); ax[1].set_xticklabels(DR)
ax[1].xaxis.set_minor_formatter(plt.NullFormatter())
ax[1].set_xlabel("dimension $d$"); ax[1].set_ylabel("ESS / gradient")
ax[1].grid(alpha=0.3, which="both"); ax[1].legend(fontsize=8.5, loc="lower left")
ax[1].set_title("(b) Rosenbrock vs $d$  (hollow = unconverged)")
ax[1].annotate("emcee fails to converge\n($\\hat R$: $1.14\\!\\to\\!2.0$)", xy=(24, 1.3e-5),
               xytext=(8.5, 2.0e-4), fontsize=8.5, color="#27ae60",
               arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.2))
fig.suptitle("Conditioning vs curvature: peaches handles both; emcee fails curvature, "
             "un-adapted NUTS fails conditioning", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIGS / "hwm_conditioning.png", dpi=150, bbox_inches="tight")
print("wrote", FIGS / "hwm_conditioning.png")
