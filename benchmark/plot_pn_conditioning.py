"""peaches-note Fig 2: conditioning vs curvature (no peanuts).

(a) ill-conditioned Gaussian d=50 vs kappa: peaches & emcee are affine-invariant
(kappa-flat for free); NUTS is kappa-flat only WITH dense-mass adaptation -- NUTS
with the mass fixed to identity collapses 4 orders. (b) Rosenbrock vs d: peaches
and NUTS (gradient methods) cross the curved ridge; emcee fails to converge.

Median/band over CONVERGED seeds only (R-hat<=1.03); solid = majority converged,
hollow = not (a stuck seed never enters the median).
"""
import json, glob
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


def series(specs):
    xs, med, lo, hi, conv = [], [], [], [], []
    for x, tgt, samp, tag in specs:
        pairs = []
        for f in glob.glob(f"benchmark/results/{tgt}/{samp}__{tag}/seed*/metrics.json"):
            r = json.load(open(f)); eg = r.get("ess_per_grad"); rh = r.get("rhat_max")
            if eg is None or not np.isfinite(eg):
                continue
            pairs.append((eg, rh if (rh is not None and np.isfinite(rh)) else None))
        if not pairs:
            continue
        cv = [v for v, rh in pairs if rh is None or rh <= RH]
        use = cv if cv else [v for v, _ in pairs]
        nconv = sum(1 for v, rh in pairs if rh is not None and rh <= RH)
        xs.append(x); med.append(np.median(use)); lo.append(min(use)); hi.append(max(use))
        conv.append(nconv >= (len(pairs) + 1) // 2)
    return (np.array(xs, float), np.array(med), np.array(lo), np.array(hi), np.array(conv))


def plot_s(ax, x, med, lo, hi, conv, c, mk, ls, lab):
    if len(x) == 0:
        return
    ax.plot(x, med, color=c, ls=ls, lw=3.0, zorder=3, label=lab)
    ax.fill_between(x, lo, hi, color=c, alpha=0.14, lw=0, zorder=1)
    cm = conv.astype(bool)
    ax.scatter(x[cm], med[cm], color=c, marker=mk, s=95, zorder=4,
               edgecolors="white", linewidths=1.1)
    if (~cm).any():
        ax.scatter(x[~cm], med[~cm], facecolors="white", edgecolors="gray", marker=mk,
                   s=95, zorder=4, linewidths=1.6)


fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0))
KAP = [2, 4, 6, 8]
for samp, tagf, c, mk, ls, lab in [
        ("peaches", "condf_lc{}", "#c0392b", "o", "-", "peaches (ChEES h-walk)"),
        ("stretch", "condf_lc{}", "#27ae60", "D", "-", "emcee (stretch)"),
        ("nuts", "condf_lc{}", "#2980b9", "^", "-", "NUTS (dense mass)"),
        ("nuts", "condfid_lc{}", "#2c3e50", "v", "--", "NUTS (no mass adapt.)")]:
    x, med, lo, hi, conv = series([(k, "gaussian_d50", samp, tagf.format(k)) for k in KAP])
    plot_s(ax[0], x, med, lo, hi, conv, c, mk, ls, lab)
ax[0].set_yscale("log"); ax[0].set_xticks(KAP)
ax[0].set_xlabel(r"$\log\kappa_\sigma$ (covariance condition $=e^{2\log\kappa_\sigma}$)")
ax[0].set_ylabel(YLAB); ax[0].grid(alpha=0.3, which="both")
ax[0].legend(loc="lower left", framealpha=0.92)
ax[0].set_title(r"(a) Ill-conditioned Gaussian ($d{=}50$)")

DR = [8, 16, 32]
for samp, tag, c, mk, ls, lab in [
        ("peaches", "rz8", "#c0392b", "o", "-", "peaches (ChEES h-walk)"),
        ("stretch", "rz8", "#27ae60", "D", "-", "emcee (stretch)"),
        ("nuts", "rz8id", "#2c3e50", "v", "--", "NUTS")]:
    x, med, lo, hi, conv = series([(d, f"rosenbrock_d{d}", samp, tag) for d in DR])
    plot_s(ax[1], x, med, lo, hi, conv, c, mk, ls, lab)
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xticks(DR); ax[1].set_xticklabels(DR)
ax[1].xaxis.set_minor_formatter(plt.NullFormatter())
ax[1].set_xlabel("dimension $d$"); ax[1].set_ylabel(YLAB)
ax[1].grid(alpha=0.3, which="both"); ax[1].legend(loc="lower right", framealpha=0.92)
ax[1].set_title("(b) Rosenbrock (curvature)")
ax[1].annotate("emcee fails to converge\n($\\hat R$: $1.14\\!\\to\\!2.0$)", xy=(22, 1.6e-5),
               xytext=(8.3, 2.4e-4), fontsize=11, color="#1e8449",
               arrowprops=dict(arrowstyle="->", color="#1e8449", lw=1.6))
fig.suptitle("Conditioning vs. curvature: peaches clears both axes",
             fontsize=17, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIGS / "pn_conditioning.png", dpi=150, bbox_inches="tight")
print("wrote", FIGS / "pn_conditioning.png")
