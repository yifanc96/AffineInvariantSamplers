"""Characterize the geometry of each astro posterior.

Given the `stretch` samples (which have R̂ ≈ 1 on all our problems, so they
represent the posterior faithfully), compute:

  - Condition number κ(C) of the empirical covariance — anisotropy.
  - Max off-diagonal |correlation| — parameter degeneracies.
  - Skewness & excess kurtosis per dim — non-Gaussianity.
  - Multimodality score: for each 1-D marginal, count KDE peaks; for each
    pair, report 2-D non-convexity via BIC(GMM k=2) − BIC(GMM k=1).
  - "Curved geometry" score: ratio of posterior std of log|∇² log p| at
    samples vs its mean — high ratio ⇒ position-dependent curvature (the
    thing that breaks HMC's single-step-size assumption).

Usage:
    python -m astro.characterize
    python -m astro.characterize --problem transit
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_stretch(problem: str, root: Path):
    # Prefer seed-aggregated stretch samples (R̂ clean).
    files = sorted(root.glob(f"{problem}/stretch/seed*/samples.npz"))
    if not files:
        files = [root / problem / "stretch" / "samples.npz"]
    arrs = []
    for f in files:
        if f.exists():
            a = np.load(f)["samples"].astype(np.float32)      # (N, C, D)
            # Discard 10% burnin per seed, flatten walkers
            a = a[a.shape[0] // 10:].reshape(-1, a.shape[-1])
            arrs.append(a)
    if not arrs:
        return None
    return np.concatenate(arrs, axis=0)                      # (T, D)


def _kde_modes_1d(x, n_grid=400):
    """Count KDE peaks in 1-D via simple finite-difference peak finding.
    Uses Silverman's rule for bandwidth, subsamples to 5k for speed."""
    x = x[np.random.default_rng(0).choice(x.size, min(5000, x.size),
                                           replace=False)]
    bw = 1.06 * x.std() * len(x) ** (-1 / 5.0)
    if bw <= 0: return 1
    g = np.linspace(x.min(), x.max(), n_grid)
    d = np.exp(-0.5 * ((g[:, None] - x[None, :]) / bw) ** 2).sum(axis=1)
    d /= d.max()
    # Peak: local max above 5% of global peak
    peaks = 0
    for i in range(1, n_grid - 1):
        if d[i] > 0.05 and d[i] > d[i-1] and d[i] > d[i+1]:
            peaks += 1
    return max(peaks, 1)


def _gmm_bic_gap(xy):
    """2-D BIC gap: BIC(k=1) − BIC(k=2).  Positive ⇒ 2 components preferred
    ⇒ bimodal or curved (non-Gaussian)."""
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        return float("nan")
    n = min(3000, xy.shape[0])
    sub = xy[np.random.default_rng(1).choice(xy.shape[0], n, replace=False)]
    try:
        bic1 = GaussianMixture(1, covariance_type="full",
                                reg_covar=1e-4).fit(sub).bic(sub)
        bic2 = GaussianMixture(2, covariance_type="full",
                                reg_covar=1e-4).fit(sub).bic(sub)
        return float(bic1 - bic2)
    except Exception:
        return float("nan")


def characterize(samples: np.ndarray):
    # Float64 for numerical stability: period-like params have std ≪ mean,
    # so (samples − mu)/std in float32 suffers catastrophic subtraction and
    # poisons skew/kurt.  Computing moments in float64 fixes it.
    samples = samples.astype(np.float64)
    T, D = samples.shape
    mu   = samples.mean(axis=0)
    C    = np.cov(samples.T)
    if D == 1:
        C = np.array([[float(C)]])
    eig  = np.sort(np.linalg.eigvalsh(C))[::-1]
    cond = float(eig[0] / max(eig[-1], 1e-30))

    # Correlation matrix, max off-diagonal absolute.
    std = np.sqrt(np.maximum(np.diag(C), 1e-30))
    R   = C / np.outer(std, std)
    np.fill_diagonal(R, 0.0)
    max_abs_corr = float(np.max(np.abs(R)))
    # Find the pair
    pair_flat = int(np.argmax(np.abs(R)))
    pair = (int(pair_flat // D), int(pair_flat % D))

    # Normality per dim
    z = (samples - mu) / std
    skew = np.mean(z ** 3, axis=0)
    kurt = np.mean(z ** 4, axis=0) - 3.0
    max_abs_skew = float(np.max(np.abs(skew)))
    max_abs_kurt = float(np.max(np.abs(kurt)))

    # Multimodality: per-dim 1-D peaks + 2-D BIC gap on strongest-corr pair
    n_modes = [_kde_modes_1d(samples[:, d]) for d in range(D)]
    bic_gap = _gmm_bic_gap(samples[:, list(pair)]) if max_abs_corr > 0.1 else 0.0

    # "Curvature-heterogeneity" proxy: how much does |posterior cov| depend on
    # where you are?  Split samples by first-PC quartiles, compute local cov
    # condition number per quartile, report ratio max/min.
    v1 = np.linalg.eigh(C)[1][:, -1]
    proj = samples @ v1
    q = np.quantile(proj, [0.25, 0.5, 0.75])
    buckets = [samples[proj <= q[0]],
               samples[(proj > q[0]) & (proj <= q[1])],
               samples[(proj > q[1]) & (proj <= q[2])],
               samples[proj > q[2]]]
    local_stds = []
    for b in buckets:
        if b.shape[0] < max(20, 2 * D):
            continue
        local_stds.append(b.std(axis=0))
    if len(local_stds) >= 2:
        ls = np.stack(local_stds, axis=0)              # (K, D)
        curv_het = float((ls.max(axis=0) / np.maximum(ls.min(axis=0), 1e-30)).max())
    else:
        curv_het = float("nan")

    return dict(
        n_samples=int(T), dim=int(D),
        mean=mu.tolist(),
        std_per_dim=std.tolist(),
        eig_top3=eig[:3].tolist(),
        eig_bot3=eig[-3:].tolist(),
        cond=cond,
        max_abs_corr=max_abs_corr,
        max_corr_pair=pair,
        max_abs_skew=max_abs_skew,
        max_abs_kurt=max_abs_kurt,
        skew_per_dim=skew.tolist(),
        kurt_per_dim=kurt.tolist(),
        n_modes_per_dim=n_modes,
        multimodal=any(m > 1 for m in n_modes),
        bic_gap_strongest_pair=bic_gap,
        curvature_heterogeneity=curv_het,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default=None)
    ap.add_argument("--results-dir", default="astro/results")
    args = ap.parse_args()

    root = Path(args.results_dir)
    problems = [args.problem] if args.problem else \
               ["hogg", "pantheon", "transit", "gw_pop"]

    print(f"{'problem':<10s} {'D':>3s}  {'κ(C)':>9s}  {'max|ρ|':>7s}  "
          f"{'|skew|':>7s}  {'|kurt|':>7s}  {'modes':>5s}  "
          f"{'BIC₁₂':>7s}  {'curv-het':>8s}")
    print("-" * 80)
    for p in problems:
        s = _load_stretch(p, root)
        if s is None:
            print(f"{p:<10s}  (no stretch samples)")
            continue
        r = characterize(s)
        modes_summary = f"{max(r['n_modes_per_dim'])}/{r['dim']}"
        print(f"{p:<10s} {r['dim']:>3d}  {r['cond']:>9.1f}  "
              f"{r['max_abs_corr']:>7.3f}  "
              f"{r['max_abs_skew']:>7.2f}  {r['max_abs_kurt']:>7.2f}  "
              f"{modes_summary:>5s}  {r['bic_gap_strongest_pair']:>7.1f}  "
              f"{r['curvature_heterogeneity']:>8.2f}")

        # Detail: highest-corr pair, and any multimodal dims.
        i, j = r["max_corr_pair"]
        print(f"  strongest corr: dim{i}×dim{j}  ρ = {r['max_abs_corr']:.3f}")
        mm_dims = [d for d, m in enumerate(r["n_modes_per_dim"]) if m > 1]
        if mm_dims:
            print(f"  multimodal dims: {mm_dims}  (modes: "
                  f"{[r['n_modes_per_dim'][d] for d in mm_dims]})")
        top_skew = int(np.argmax(np.abs(r["skew_per_dim"])))
        top_kurt = int(np.argmax(np.abs(r["kurt_per_dim"])))
        print(f"  max |skew|: dim{top_skew} = {r['skew_per_dim'][top_skew]:+.2f}   "
              f"max |kurt|: dim{top_kurt} = {r['kurt_per_dim'][top_kurt]:+.2f}")


if __name__ == "__main__":
    main()
