"""Option B corner plot for v2 paper figure: 8 dims (Ω_c, σ_8, A, b_1..b_5).

Shows the slow subspace structure: cosmology recovery (Ω_c, σ_8, A near
truth) coexists with multiple narrow valleys (the σ_8↔b_i and b_i↔b_j
ridges from PC analysis) that drive the per-dim ESS bottleneck.

Uses clean PICKLES γ=0.5 ws=200 prior_median samples (3 seeds pooled).
Output: figs/des_y1_corner.pdf, figs/des_y1_corner.png.
"""
from __future__ import annotations
import sys
sys.path.insert(0, '/home/yifanchen/research/AffineInvariantSamplers')

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json

from astro.likelihoods_des_y1 import _UNIF, _UNIF_LO, _UNIF_HI, _NORM, _FID

ROOT = Path("/home/yifanchen/research/AffineInvariantSamplers/astro")
RUN_DIR = ROOT / "results/des_y1/pickles_priormed_g05_ws200"
FIG_DIR = ROOT / "figs"
FIG_DIR.mkdir(exist_ok=True)


# ─── z-space → physical-space conversion ────────────────────────────────────
def z_to_phys(z):
    """z-space (logit/standard-normal) → physical params, shape preserved."""
    n_u = len(_UNIF)
    out = np.zeros_like(z)
    # Uniform priors: x = lo + (hi - lo) * sigmoid(z)
    out[..., :n_u] = _UNIF_LO + (_UNIF_HI - _UNIF_LO) / (1 + np.exp(-z[..., :n_u]))
    # Normal priors (m, dz): non-centered → x = mu + sd * z_decentered
    means = np.array([m for _, m, _ in _NORM])
    stds  = np.array([s for _, _, s in _NORM])
    out[..., n_u:] = means + stds * z[..., n_u:]
    return out


# ─── Load PICKLES samples (3 seeds pooled) ─────────────────────────────────
def load_samples():
    pieces = []
    for seed in [0, 1, 2]:
        f = RUN_DIR / f"seed{seed}/samples.npz"
        if not f.exists():
            print(f"  skip {f} (missing)")
            continue
        x = np.load(f)["samples"].astype(np.float64)   # (N, C, 21) z-space
        N, C, D = x.shape
        # Drop first 200 burn-in samples per chain
        x = x[200:]
        flat = x.reshape(-1, D)
        pieces.append(flat)
        print(f"  loaded seed{seed}: {flat.shape}")
    samples_z = np.concatenate(pieces, axis=0)
    print(f"pooled: {samples_z.shape}")
    return z_to_phys(samples_z)


# ─── Build the 8×8 corner plot ──────────────────────────────────────────────
PARAM_NAMES = (["Omega_c", "sigma8", "Omega_b", "h", "n_s", "w0",
                 "A", "eta",
                 "b1", "b2", "b3", "b4", "b5",
                 "m1", "m2", "m3", "m4",
                 "dz1", "dz2", "dz3", "dz4"])

PRETTY = {
    "Omega_c": r"$\Omega_c$", "sigma8": r"$\sigma_8$",
    "n_s": r"$n_s$",
    "A": r"$A$", "eta": r"$\eta$",
    "b1": r"$b_1$", "b2": r"$b_2$", "b3": r"$b_3$", "b4": r"$b_4$", "b5": r"$b_5$",
}

# Option B subspace (8 dims: cosmology recovery + full bias subspace)
DIMS_OPT_B = ["Omega_c", "sigma8", "A", "b1", "b2", "b3", "b4", "b5"]
# Option C — two separate 6×6 plots
DIMS_OPT_C1 = ["Omega_c", "sigma8", "n_s", "A", "b1", "b2"]   # cosmology-style
DIMS_OPT_C2 = ["sigma8",  "b1",     "b2",  "b3", "b4", "b5"]  # slow subspace


def make_corner_v2(samples, params=DIMS_OPT_B, out_stem="des_y1_corner",
                    title_extra=None):
    idx = [PARAM_NAMES.index(p) for p in params]
    D = len(params)
    data = samples[:, idx]
    truths = np.array([_FID["b"][int(p[1:]) - 1] if p.startswith("b") else _FID[p]
                        for p in params])

    # Subsample to 30k for plot speed
    rng = np.random.default_rng(0)
    sub = rng.choice(data.shape[0], min(30_000, data.shape[0]), replace=False)
    d = data[sub]

    fig, axes = plt.subplots(D, D, figsize=(2.2 * D, 2.2 * D))

    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            if i == j:
                # Diagonal: marginal histogram
                ax.hist(d[:, i], bins=50, color="#3b6ea5", density=True,
                         alpha=0.85)
                ax.set_yticks([])
                ax.set_title(f"{PRETTY.get(params[i], params[i])}",
                             fontsize=14, pad=3)
            elif i > j:
                # Lower triangle: 2D density + ρ
                ax.hexbin(d[:, j], d[:, i], gridsize=40, cmap="Blues",
                           mincnt=1)
                rho = np.corrcoef(d[:, j], d[:, i])[0, 1]
                ax.text(0.05, 0.92, fr"$\rho={rho:+.2f}$",
                         transform=ax.transAxes, fontsize=11,
                         color="#c0392b" if abs(rho) > 0.5 else "k",
                         bbox=dict(facecolor="white", edgecolor="none",
                                   alpha=0.8, pad=1.5))
            else:
                # Upper triangle: blank
                ax.axis("off")

            # Tick labels
            if i == D - 1 and i != j:
                ax.set_xlabel(PRETTY.get(params[j], params[j]), fontsize=14)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(PRETTY.get(params[i], params[i]), fontsize=14)
            elif i > 0:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=10)

    title = (title_extra if title_extra else
              "DES Y1 $3\\times 2$pt posterior")
    fig.suptitle(title, fontsize=18, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out_pdf = FIG_DIR / f"{out_stem}.pdf"
    out_png = FIG_DIR / f"{out_stem}.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_pdf}\nwrote {out_png}")
    return fig


if __name__ == "__main__":
    samples = load_samples()
    # Option B — single 8×8 plot showing cosmology + full bias subspace
    make_corner_v2(samples, params=DIMS_OPT_B, out_stem="des_y1_corner_optB",
                    title_extra="DES Y1 $3\\times 2$pt posterior")
    # Option C — two separate 6×6 plots
    make_corner_v2(samples, params=DIMS_OPT_C1, out_stem="des_y1_corner_optC1",
                    title_extra="DES Y1 $3\\times 2$pt posterior — "
                                 "cosmology marginals")
    make_corner_v2(samples, params=DIMS_OPT_C2, out_stem="des_y1_corner_optC2",
                    title_extra="DES Y1 $3\\times 2$pt posterior — "
                                 "$\\sigma_8$ and galaxy biases")
