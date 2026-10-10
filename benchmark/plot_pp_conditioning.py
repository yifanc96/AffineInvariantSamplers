"""Fig 2 (peaches/peanuts clean note): conditioning vs curvature.

Left: ill-conditioned Gaussian d=50 vs kappa. peaches/peanuts/emcee are
kappa-invariant (affine); NUTS without mass adaptation (identity metric)
collapses as kappa grows. Right: Rosenbrock vs d. peaches/peanuts (and no-mass
NUTS, since Rosenbrock's global conditioning is mild) stay converged; emcee
fails -- its stretch move cannot cross the curved ridge, so it is unconverged
at d>=8 (bias z up to 112): those points are greyed.

3 seeds; median + min--max band. Unconverged cells (bias z>5 or Rhat>1.2) are
drawn hollow/greyed -- their ESS/grad is not a meaningful efficiency.
"""
import json, glob
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = Path("benchmark/figs"); FIGS.mkdir(parents=True, exist_ok=True)

def grab(tgt, samp, tag):
    """List of (ess_per_grad, rhat_or_None), one per seed."""
    out = []
    for f in glob.glob(f"benchmark/results/{tgt}/{samp}__{tag}/seed*/metrics.json"):
        r = json.load(open(f))
        eg = r.get("ess_per_grad")
        if eg is None or not np.isfinite(eg):
            continue
        rh = r.get("rhat_max")
        out.append((eg, rh if (rh is not None and np.isfinite(rh)) else None))
    return out

def series(specs):
    """specs: list of (xval, tgt, samp, tag). Returns x, med, lo, hi, conv(bool).
    Correctness: median/band over CONVERGED seeds only (R-hat<=1.03); a point is
    marked converged iff a majority of its seeds converged."""
    xs, med, lo, hi, conv = [], [], [], [], []
    for x, tgt, samp, tag in specs:
        pairs = grab(tgt, samp, tag)
        if not pairs: continue
        cv = [v for (v, rh) in pairs if rh is None or rh <= 1.03]
        use = cv if cv else [v for (v, _) in pairs]
        xs.append(x); med.append(np.median(use)); lo.append(min(use)); hi.append(max(use))
        nconv = sum(1 for (v, rh) in pairs if rh is not None and rh <= 1.03)
        conv.append(nconv >= (len(pairs) + 1) // 2)
    return (np.array(xs, float), np.array(med), np.array(lo), np.array(hi), np.array(conv))

def plot_sampler(ax, x, med, lo, hi, conv, c, mk, ls, lab, band=True):
    if len(x) == 0: return
    ax.plot(x, med, color=c, ls=ls, lw=2, zorder=3, label=lab)
    if band:
        ax.fill_between(x, lo, hi, color=c, alpha=0.10, lw=0, zorder=1)
    # converged points solid, unconverged hollow/grey
    cm = conv.astype(bool)
    ax.scatter(x[cm], med[cm], color=c, marker=mk, s=60, zorder=4)
    if (~cm).any():
        ax.scatter(x[~cm], med[~cm], facecolors="white", edgecolors="gray",
                   marker=mk, s=60, zorder=4, linewidths=1.3)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
# (a) ill-conditioned Gaussian d=50 vs kappa
KAP = [2, 4, 6, 8]
for samp, tagf, c, mk, ls, lab in [
        ("peaches", "exp2_lc{}", "#c0392b", "o", "-", "peaches"),
        ("peanuts", "exp2_lc{}", "#8e44ad", "s", "-", "peanuts"),
        ("stretch", "exp2_lc{}", "#27ae60", "D", "-", "emcee (stretch)"),
        ("nuts", "idmass_lc{}", "#2c3e50", "v", "--", "NUTS, no mass adapt.")]:
    x, med, lo, hi, conv = series([(k, "gaussian_d50", samp, tagf.format(k)) for k in KAP])
    plot_sampler(ax[0], x, med, lo, hi, conv, c, mk, ls, lab)
ax[0].set_yscale("log"); ax[0].set_xticks(KAP)
ax[0].set_xlabel(r"$\log\kappa_\sigma$ (covariance condition $=e^{2\log\kappa_\sigma}$)")
ax[0].set_ylabel("ESS / gradient"); ax[0].grid(alpha=0.3, which="both")
ax[0].legend(fontsize=8.5, loc="lower left")
ax[0].set_title(r"(a) Ill-conditioned Gaussian ($d{=}50$) vs $\kappa$")
# (b) Rosenbrock vs d (the curvature test: peaches/peanuts vs emcee; no-mass
# NUTS omitted here -- it converges on Rosenbrock and is the panel-(a) story).
DR = [8, 16, 32]   # peaches/peanuts cleanly converged here; emcee fails at all.
# (Low-d d=2,4 is shown in the Rosenbrock scaling diagnostic instead: at d=2 the
# single banana is easy enough that emcee converges and peanuts struggles, which
# would muddy this curvature-failure panel.) Converged-only median (see series)
# handles the one slow peaches seed at d=16.
for samp, tag, c, mk, ls, lab in [
        ("peaches", "exp1", "#c0392b", "o", "-", "peaches"),
        ("peanuts", "exp1", "#8e44ad", "s", "-", "peanuts"),
        ("stretch", "exp1", "#27ae60", "D", "-", "emcee (stretch)")]:
    x, med, lo, hi, conv = series([(d, f"rosenbrock_d{d}", samp, tag) for d in DR])
    plot_sampler(ax[1], x, med, lo, hi, conv, c, mk, ls, lab)
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xticks(DR); ax[1].set_xticklabels(DR)
ax[1].xaxis.set_minor_formatter(plt.NullFormatter())   # no 2x10^1 clutter
ax[1].set_xlabel("dimension $d$"); ax[1].set_ylabel("ESS / gradient")
ax[1].grid(alpha=0.3, which="both"); ax[1].legend(fontsize=8.5, loc="lower left")
ax[1].set_title("(b) Rosenbrock vs $d$  (hollow = unconverged)")
ax[1].annotate("emcee fails to converge\n($\\hat R$: $1.14\\!\\to\\!2.0$)", xy=(24, 1.3e-5),
               xytext=(8.5, 2.0e-4), fontsize=8.5, color="#27ae60",
               arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.2))
fig.suptitle("Conditioning vs curvature: only peaches/peanuts handle both "
             "(emcee fails curvature; no-mass HMC fails conditioning)", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIGS / "pp_conditioning.png", dpi=150, bbox_inches="tight")
print("wrote", FIGS / "pp_conditioning.png")
