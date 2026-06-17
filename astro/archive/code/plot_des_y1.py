"""Figures for the DES Y1 3×2pt benchmark.

Produces two PDFs / PNGs:

  figs/des_y1_corner.png       — corner plot of key parameters from the
                                  peaches samples, showing the S_8 banana
                                  + intrinsic-alignment degeneracies that
                                  make this a hard inference problem.
  figs/des_y1_leaderboard.png  — horizontal bar chart: ESS/sec per sampler,
                                  log-x, annotated with per-gradient efficiency.

Usage::

    python -m astro.plot_des_y1
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astro.likelihoods_des_y1 import (_UNIF, _NORM, _z_to_physical,
                                       _UNIF_LO, _UNIF_HI,
                                       _NORM_MU,  _NORM_SD)


_FIGDIR = Path(__file__).parent / "figs"
_FIGDIR.mkdir(exist_ok=True)


# ─── Corner plot from peaches samples ────────────────────────────────────────

def _load_peaches_phys():
    import jax.numpy as jnp
    files = sorted(Path("astro/results/des_y1/peaches").glob("seed*/samples.npz"))
    arrs = []
    for f in files:
        a = np.load(f)["samples"].astype(np.float32)        # (N, C, D)
        a = a[a.shape[0] // 10:].reshape(-1, a.shape[-1])
        arrs.append(a)
    z = np.concatenate(arrs, axis=0)
    # z-space → physical.  Avoid JIT; vectorised numpy.
    n_u = len(_UNIF)
    zu, zn = z[:, :n_u], z[:, n_u:]
    sig = 1.0 / (1.0 + np.exp(-zu))
    phys_u = np.asarray(_UNIF_LO) + (np.asarray(_UNIF_HI) - np.asarray(_UNIF_LO)) * sig
    phys_n = np.asarray(_NORM_MU) + np.asarray(_NORM_SD) * zn
    phys = np.concatenate([phys_u, phys_n], axis=1)
    names = [n for n, _, _ in _UNIF] + [n for n, _, _ in _NORM]
    return phys, names


def _derive_S8(phys, names):
    Om = phys[:, names.index("Omega_c")] + phys[:, names.index("Omega_b")]
    sig8 = phys[:, names.index("sigma8")]
    S8 = sig8 * np.sqrt(Om / 0.3)
    return Om, sig8, S8


def make_corner():
    phys, names = _load_peaches_phys()
    Om, sig8, S8 = _derive_S8(phys, names)

    # Choose 6 params that tell the story: Ω_m (derived), σ_8, S_8 (derived),
    # A_IA, η_IA, b_3 (galaxy bias, middle lens bin).
    cols = [("Ω_m  (Ω_c+Ω_b)", Om),
            ("σ_8",            sig8),
            ("S_8  =  σ_8·√(Ω_m/0.3)", S8),
            ("A_IA",            phys[:, names.index("A")]),
            ("η_IA",            phys[:, names.index("eta")]),
            ("b_3 (galaxy bias)", phys[:, names.index("b3")])]
    D = len(cols)

    # Subsample to 20k for plot speed.
    idx = np.random.default_rng(0).choice(phys.shape[0],
                                           min(20_000, phys.shape[0]),
                                           replace=False)
    data = np.stack([c[1][idx] for c in cols], axis=1)
    labels = [c[0] for c in cols]

    fig, axes = plt.subplots(D, D, figsize=(12, 12))
    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            if i == j:
                ax.hist(data[:, i], bins=60, color="C0", density=True,
                         alpha=0.9)
                ax.set_yticks([])
            elif i > j:
                ax.hexbin(data[:, j], data[:, i], gridsize=40, cmap="Blues",
                          mincnt=1)
            else:
                ax.axis("off")
            if i == D - 1:
                ax.set_xlabel(labels[j], fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i], fontsize=8)
            elif i > 0:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)

    fig.suptitle("DES Y1 3×2pt posterior (peaches, 3 seeds pooled) — "
                 "21D cosmology + nuisance, showing the S_8 banana and "
                 "A_IA/η_IA degeneracies", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = _FIGDIR / "des_y1_corner.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


# ─── Leaderboard bar chart ──────────────────────────────────────────────────

def _collect_cells():
    """Return {sampler: {ess_min, ess_sec, ess_eval, eval_kind}} averaged across seeds."""
    out = {}
    for sampler_dir in sorted(Path("astro/results/des_y1").iterdir()):
        if not sampler_dir.is_dir(): continue
        metrics = []
        for s in sampler_dir.glob("seed*/metrics.json"):
            metrics.append(json.loads(s.read_text()))
        if not metrics: continue
        ess_min = np.median([m["ess_min"] for m in metrics])
        ess_sec = np.median([m["ess_per_sec"] for m in metrics])
        # For chess, ess_per_eval was computed with n_lp_evals pre-fix;
        # recompute from info + nominal_L.
        ess_evals = []
        kinds = []
        for m in metrics:
            ne = m.get("n_target_evals") or 0
            if m["sampler"] == "chess":
                L = m["info"].get("nominal_L", 1)
                iters = (m["num_samples"] + m["warmup"]) * m["n_chains"]
                ne = iters * int(L)
                kinds.append("grad")
            else:
                kinds.append(m.get("eval_kind", "??"))
            if ne:
                ess_evals.append(m["ess_min"] / ne)
        ess_eval = np.median(ess_evals) if ess_evals else np.nan
        out[sampler_dir.name] = dict(
            ess_min=ess_min, ess_sec=ess_sec, ess_eval=ess_eval,
            eval_kind=kinds[0] if kinds else "?")
    return out


def make_leaderboard():
    cells = _collect_cells()
    order = ["stretch", "chess", "pickles", "peams", "malt",
              "nuts_numpyro_diag", "nuts_numpyro_block",
              "nuts_numpyro_dense", "peanuts", "peaches"]
    cells = {k: v for k, v in cells.items() if k in order}
    order = [k for k in order if k in cells]
    names = order
    ess_sec = [cells[n]["ess_sec"] for n in names]
    ess_eval = [cells[n]["ess_eval"] for n in names]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = ["#999"] * len(names)
    for i, n in enumerate(names):
        if "peaches" in n or "peanuts" in n or "peams" in n or "pickles" in n:
            colors[i] = "#c0392b"       # affine-invariant HMC — red
        elif "chess" in n:
            colors[i] = "#f39c12"       # identity-mass HMC — orange
        elif "nuts" in n:
            colors[i] = "#27ae60"       # NUTS — green
        elif n == "stretch" or n == "emcee":
            colors[i] = "#2980b9"       # affine-invariant gradient-free — blue

    axL.barh(names, ess_sec, color=colors, edgecolor="k")
    axL.set_xscale("log")
    axL.set_xlabel("ess_min / wall-second")
    axL.set_title("Wall-time efficiency")
    axL.grid(axis="x", ls=":", alpha=0.5)

    axR.barh(names, ess_eval, color=colors, edgecolor="k")
    axR.set_xscale("log")
    axR.set_xlabel("ess_min / gradient (or density) evaluation")
    axR.set_title("Per-evaluation efficiency (AD-fair)")
    axR.grid(axis="x", ls=":", alpha=0.5)

    # Annotate with ess_min value
    for ax, vals in [(axL, ess_sec), (axR, ess_eval)]:
        for i, v in enumerate(vals):
            if not np.isfinite(v): continue
            ax.text(v * 1.05, i, f"{v:.2g}", va="center", fontsize=8)

    fig.suptitle("DES Y1 3×2pt — 21D cosmology + nuisance benchmark",
                  fontsize=11, y=1.02)
    fig.tight_layout()
    out = _FIGDIR / "des_y1_leaderboard.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


# ─── 2×2 "science overview" panel ────────────────────────────────────────────

def make_overview():
    """Four-panel visual explanation of the DES Y1 3×2pt inverse problem.

    (A) The n(z) redshift distributions for the 4 source + 5 lens tomographic
        bins — what "tomography" means on a galaxy catalog.
    (B) Angular correlation functions ξ_+(θ) for one representative bin pair
        — what the *observables* are.  Theory curves at fiducial cosmology.
    (C) Full 457-element data vector with diagonal ±1σ from the covariance —
        what the "data" actually is, and how noise-dominated each entry is.
    (D) 2D posterior Ω_m–σ_8 recovered by peaches, showing the S_8 banana and
        the fiducial truth as a cross.
    """
    import os
    from astropy.io import fits
    import jax_cosmo as jc
    import jax.numpy as jnp
    from astro.likelihoods_des_y1 import _theory_and_data, _FID

    data_path = os.path.join(os.path.dirname(__file__), "data",
                              "2pt_NG_mcal_1110.fits")
    with fits.open(data_path) as hdul:
        nz_source = hdul[6].data
        nz_lens   = hdul[7].data
    z_s = nz_source["Z_MID"].astype(float)
    z_l = nz_lens["Z_MID"].astype(float)

    print("[overview] Computing fiducial theory + cov …")
    theory_fn, data_mean, cov, fid_phys = _theory_and_data()
    data_mean = np.asarray(data_mean)
    cov_np = np.asarray(cov)
    sigma = np.sqrt(np.maximum(np.diag(cov_np), 0.0))

    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)

    # ── Panel A: n(z) ──
    axA = fig.add_subplot(gs[0, 0])
    colors_s = plt.cm.viridis(np.linspace(0.05, 0.8, 4))
    colors_l = plt.cm.plasma (np.linspace(0.05, 0.8, 5))
    for i, c in enumerate(colors_s, start=1):
        axA.fill_between(z_s, nz_source[f"BIN{i}"], alpha=0.4, color=c,
                          label=f"source bin {i}")
    for i, c in enumerate(colors_l, start=1):
        axA.plot(z_l, nz_lens[f"BIN{i}"], color=c, lw=1.8, ls="--",
                  label=f"lens bin {i}")
    axA.set_xlim(0, 1.8)
    axA.set_xlabel("redshift  z")
    axA.set_ylabel("n(z)  (unnormalised)")
    axA.set_title("(A)  Tomographic galaxy samples — 4 source + 5 lens bins")
    axA.legend(ncol=3, fontsize=7, loc="upper right")

    # ── Panel B: cartoon correlation function at fiducial Λ ──
    axB = fig.add_subplot(gs[0, 1])
    # The data vector is concatenation of 4 probes at 50 ells.
    # For the story, show the first block (shear ξ_+ in bin pair (1,1)).
    # jax-cosmo's angular_cl returns (n_probes, n_ell) so we need to reconstruct.
    ell = np.logspace(1, 3, 50)
    # For a "what does the cosmic shear signal look like" story, show the first
    # 50 elements of the data vector as ell·C_ell (proxy for signal shape).
    block_len = 50
    for k, (lbl, col) in enumerate([("shear + clustering + lensing\n"
                                      "first tomographic block", "C0")]):
        d = data_mean[:block_len]
        s = sigma[:block_len]
        axB.loglog(ell, d * ell, color=col, lw=2, label=lbl)
        axB.fill_between(ell, (d - s) * ell, (d + s) * ell,
                          color=col, alpha=0.25)
    axB.set_xlabel(r"multipole  $\ell$")
    axB.set_ylabel(r"$\ell \cdot C_\ell$  (first block)")
    axB.set_title("(B)  Angular power spectrum — 1 of 9 tomographic blocks")
    axB.grid(alpha=0.3)
    axB.legend(fontsize=8, loc="lower left")

    # ── Panel C: full data vector ──
    axC = fig.add_subplot(gs[1, 0])
    ix = np.arange(data_mean.size)
    axC.errorbar(ix, data_mean, yerr=sigma, fmt=".", ms=1.5,
                  ecolor="0.7", mfc="C0", mec="C0", alpha=0.8)
    axC.set_xlabel("index into 3×2pt data vector")
    axC.set_ylabel("value  (angular power spectrum)")
    axC.set_title(f"(C)  Full data vector: {data_mean.size} entries, "
                   f"dense covariance")
    axC.set_yscale("symlog", linthresh=1e-10)
    axC.grid(alpha=0.3)

    # ── Panel D: peaches posterior on Ω_m–σ_8 with ρ & contours ──
    axD = fig.add_subplot(gs[1, 1])
    try:
        phys, names = _load_peaches_phys()
        Om = phys[:, names.index("Omega_c")] + phys[:, names.index("Omega_b")]
        sig8 = phys[:, names.index("sigma8")]
        rho = float(np.corrcoef(Om, sig8)[0, 1])
        idx = np.random.default_rng(0).choice(phys.shape[0],
                                               min(30_000, phys.shape[0]),
                                               replace=False)
        xs, ys = Om[idx], sig8[idx]
        axD.hexbin(xs, ys, gridsize=50, cmap="Reds", mincnt=1)
        # Overlay 1σ/2σ iso-density contours via a 2-D histogram.
        from scipy.stats import gaussian_kde
        H, xe, ye = np.histogram2d(xs, ys, bins=60)
        H = H / H.sum()
        hs = np.sort(H.ravel())[::-1]
        cs = np.cumsum(hs)
        lvls = [hs[np.searchsorted(cs, q)] for q in (0.683, 0.954)][::-1]
        X, Y = np.meshgrid(0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]),
                            indexing="ij")
        axD.contour(X, Y, H, levels=lvls, colors=["k", "k"],
                     linewidths=[1.6, 0.9], linestyles=["-", "--"])
        # Annotate correlation + banana factor
        sig8_std = sig8.std(); S8 = sig8 * np.sqrt(Om / 0.3); S8_std = S8.std()
        axD.text(0.03, 0.97,
                  f"ρ(Ω_m, σ_8) = {rho:+.2f}\n"
                  f"σ(σ_8) / σ(S_8) = {sig8_std/S8_std:.2f}\n"
                  f"(mock data — weaker banana than real DES Y1)",
                  transform=axD.transAxes, fontsize=7.5, va="top",
                  bbox=dict(boxstyle="round", fc="white", ec="0.6"))
        xlo, xhi = np.percentile(Om, [0.5, 99.5])
        ylo, yhi = np.percentile(sig8, [0.5, 99.5])
        pad = 0.15
        axD.set_xlim(xlo - pad * (xhi - xlo), xhi + pad * (xhi - xlo))
        axD.set_ylim(ylo - pad * (yhi - ylo), yhi + pad * (yhi - ylo))
    except Exception as e:
        axD.text(0.5, 0.5, f"(peaches samples missing: {e})",
                  ha="center", va="center", transform=axD.transAxes)
    # Draw S_8 = σ_8·√(Ω_m/0.3) iso-contours for reference
    Om_grid = np.linspace(0.05, 0.7, 100)
    for S8_val in [0.55, 0.65, 0.75, 0.85]:
        sig_iso = S8_val / np.sqrt(Om_grid / 0.3)
        axD.plot(Om_grid, sig_iso, "k:", lw=0.6, alpha=0.35)
        # label at right edge if in view
        yl = axD.get_ylim()
        y_at = S8_val / np.sqrt(axD.get_xlim()[1] / 0.3)
        if yl[0] < y_at < yl[1]:
            axD.text(axD.get_xlim()[1], y_at, f" S₈={S8_val}",
                      fontsize=6.5, va="center", color="0.4")
    # Fiducial + posterior mean markers
    Om_fid = _FID["Omega_c"] + _FID["Omega_b"]
    axD.plot([Om_fid], [_FID["sigma8"]], "k+", ms=14, mew=2,
              label="fiducial truth")
    axD.plot([Om.mean()], [sig8.mean()], "rx", ms=10, mew=2,
              label="posterior mean")
    axD.set_xlabel(r"$\Omega_m$")
    axD.set_ylabel(r"$\sigma_8$")
    axD.legend(fontsize=7.5, loc="lower left")
    axD.set_title("(D)  Peaches posterior — 21-D marginalised to (Ω_m, σ_8),\n"
                   "solid/dashed = 1σ/2σ iso-density")

    fig.suptitle(
        "DES Y1 3×2pt — the inverse problem\n"
        "Given the 4-source + 5-lens tomographic galaxy samples and their "
        f"{data_mean.size}-element two-point angular data vector with dense "
        f"{data_mean.size}×{data_mean.size} covariance,\n"
        "infer 21 parameters (6 cosmology + 17 nuisance).  High-dimensional "
        "with dense cross-parameter correlations — the HMC sweet spot.",
        fontsize=11, y=1.01)
    fig.tight_layout()
    out = _FIGDIR / "des_y1_overview.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


# ─── Contour-overlap verification (mirror Campagne+ 2023 Fig. 11) ────────────

def _load_phys(sampler: str):
    """Load and transform samples (z-space → physical) from disk."""
    import jax.numpy as jnp
    files = sorted(Path(f"astro/results/des_y1/{sampler}").glob(
                    "seed*/samples.npz"))
    if not files:
        return None, None
    arrs = []
    for f in files:
        a = np.load(f)["samples"].astype(np.float64)
        # NumPyro samples are stored in physical space already.
        # Our repo's samplers run in z-space and need the transform.
        arrs.append(a[a.shape[0] // 10:].reshape(-1, a.shape[-1]))
    z = np.concatenate(arrs, axis=0)
    if sampler.startswith("nuts_numpyro"):
        # NumPyro ran in physical space with constraints; samples are physical.
        names = [n for n, _, _ in _UNIF] + [n for n, _, _ in _NORM]
        return z, names
    # Our repo's samplers — transform z → physical.
    n_u = len(_UNIF)
    lo, hi, mu, sd = map(np.asarray, (_UNIF_LO, _UNIF_HI, _NORM_MU, _NORM_SD))
    phys_u = lo + (hi - lo) / (1 + np.exp(-z[:, :n_u]))
    phys_n = mu + sd * z[:, n_u:]
    phys = np.concatenate([phys_u, phys_n], axis=1)
    names = [n for n, _, _ in _UNIF] + [n for n, _, _ in _NORM]
    return phys, names


def _contour_levels_2d(x, y, bins=60):
    """Return (X, Y, H, levels) for the 30/68/90% HDRs of a 2-D hist."""
    H, xe, ye = np.histogram2d(x, y, bins=bins)
    H = H / H.sum()
    hs = np.sort(H.ravel())[::-1]
    cs = np.cumsum(hs)
    lvls = [hs[np.searchsorted(cs, q)] for q in (0.3, 0.68, 0.9)][::-1]
    X, Y = np.meshgrid(0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]),
                        indexing="ij")
    return X, Y, H, lvls


def make_contour_overlap(samplers=("peaches", "peanuts",
                                     "nuts_numpyro_dense"),
                          params=("Omega_c", "sigma8", "A", "b3")):
    """Triangle plot comparing N samplers' posteriors — verification that
    all samplers converge to the same distribution.  Mirrors Campagne+ 2023
    Fig. 11 (Cobaya-MH vs NumPyro-NUTS vs SVI-NeuTra).

    If the contours overlap (they should, for independent correctly-sampled
    chains targeting the same posterior), ESS-per-sampler ratios are a real
    efficiency comparison, not a mis-sampling artifact.
    """
    import matplotlib.patches as mpatches

    results = {}
    for s in samplers:
        phys, names = _load_phys(s)
        if phys is None:
            print(f"  skip {s} (no samples)")
            continue
        col_idx = [names.index(p) for p in params]
        # Subsample to 20k for speed
        rng = np.random.default_rng(0)
        idx = rng.choice(phys.shape[0], min(20_000, phys.shape[0]),
                          replace=False)
        results[s] = phys[idx][:, col_idx]

    D = len(params)
    # Color scheme — affine-invariant HMC red, ensemble-NUTS orange,
    # NumPyro-NUTS-dense green
    palette = {"peaches": "#c0392b", "peanuts": "#e67e22",
                "peams": "#f39c12",   "pickles": "#d35400",
                "nuts_numpyro_dense": "#27ae60",
                "nuts_numpyro_block": "#16a085",
                "nuts_numpyro_diag":  "#95a5a6",
                "stretch": "#2980b9"}
    colors = [palette.get(s, f"C{i}") for i, s in enumerate(results)]

    fig, axes = plt.subplots(D, D, figsize=(3.0 * D, 3.0 * D))
    pretty = {"Omega_c": r"$\Omega_c$", "sigma8": r"$\sigma_8$",
               "A": r"$A_{\rm IA}$", "b3": r"$b_3$",
               "Omega_b": r"$\Omega_b$", "h": "$h$",
               "n_s": "$n_s$", "w0": "$w_0$", "eta": r"$\eta_{\rm IA}$"}
    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            if i == j:
                for (s, data), c in zip(results.items(), colors):
                    ax.hist(data[:, i], bins=50, density=True,
                             histtype="step", color=c, lw=1.8,
                             label=s.replace("nuts_numpyro_", "NUTS-"))
                ax.set_yticks([])
                if i == 0:
                    ax.legend(fontsize=8, loc="upper right",
                               frameon=False)
            elif i > j:
                for (s, data), c in zip(results.items(), colors):
                    X, Y, H, lvls = _contour_levels_2d(data[:, j], data[:, i],
                                                        bins=50)
                    # Filled lightest at 90%, mid at 68%, dark at 30%
                    ax.contourf(X, Y, H, levels=lvls + [H.max() + 1],
                                 colors=[c + "22", c + "44", c + "77"],
                                 alpha=0.8)
                    ax.contour(X, Y, H, levels=lvls, colors=c, linewidths=0.8)
            else:
                ax.axis("off")
            if i == D - 1:
                ax.set_xlabel(pretty.get(params[j], params[j]), fontsize=10)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(pretty.get(params[i], params[i]), fontsize=10)
            elif i > 0:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=8)

    fig.suptitle(
        "DES Y1 3×2pt — posterior overlap across samplers\n"
        "30%, 68%, 90% HDR contours.  If contours agree, all samplers target "
        "the same posterior (gold-standard ESS-comparison verification).",
        fontsize=11, y=1.01)
    fig.tight_layout()
    out = _FIGDIR / "des_y1_contour_overlap.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    make_overview()
    make_corner()
    make_leaderboard()
    make_contour_overlap()
