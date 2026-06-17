"""Cross-seed convergence diagnostics.

High per-seed ESS is not enough — a sampler can have excellent within-mode
mixing but miss modes entirely.  The only honest diagnostic for this is
cross-chain / cross-seed comparison of the marginals.

For each (problem, sampler) cell we load the 3 per-seed sample files
(``astro/results/<problem>/<sampler>/seed<k>/samples.npz``) and compute:

  - Gelman-Rubin R̂ per dimension, treating each seed as one chain (walkers
    flattened).  R̂ ≈ 1 means seeds agree on the marginal; R̂ > 1.1 is a
    red flag for multimodality or seed-dependent trapping.
  - Per-dim 1-D Wasserstein between seed-pairs, reported as median.  This
    is modality-aware: if two seeds explored different modes, the 1-D
    Wasserstein between their marginals is O(mode separation), not O(std).

Usage:
    python -m astro.convergence
    python -m astro.convergence --problem transit --sampler peaches
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np


def _rhat(chains):
    """Gelman-Rubin R̂ per dim.  `chains` has shape (M, N, D)."""
    M, N, D = chains.shape
    within  = chains.var(axis=1, ddof=1).mean(axis=0)           # (D,)
    between = chains.mean(axis=1).var(axis=0, ddof=1) * N       # (D,)
    var_plus = ((N - 1) / N) * within + between / N
    return np.sqrt(var_plus / np.maximum(within, 1e-300))


def _wass1d(a, b, n=2000):
    """1-D Wasserstein via quantile matching, subsampled."""
    a = np.sort(np.random.default_rng(0).choice(a, size=min(n, a.size),
                                                 replace=False))
    b = np.sort(np.random.default_rng(1).choice(b, size=min(n, b.size),
                                                 replace=False))
    if a.size != b.size:
        k = min(a.size, b.size)
        a, b = a[:k], b[:k]
    return float(np.mean(np.abs(a - b)))


def _load_seed_samples(problem: str, sampler: str, root: Path):
    seeds = sorted(root.glob(f"{problem}/{sampler}/seed*/samples.npz"))
    out = []
    for s in seeds:
        arr = np.load(s)["samples"].astype(np.float32)         # (N, C, D)
        out.append(arr)
    if not out:
        return None
    # Discard burnin (first 10% of each seed)
    out = [a[a.shape[0] // 10:] for a in out]
    return out


def analyze(problem: str, sampler: str, root: Path):
    arrs = _load_seed_samples(problem, sampler, root)
    if not arrs or len(arrs) < 2:
        return None

    # Per-seed flat chains: shape (M, N*C, D) → (M, T, D).
    flat = [a.reshape(-1, a.shape[-1]) for a in arrs]
    T = min(f.shape[0] for f in flat)
    chains = np.stack([f[:T] for f in flat], axis=0)          # (M, T, D)

    rhat = _rhat(chains)

    # Pairwise per-dim Wasserstein, normalised by the pooled std so it's
    # comparable across dimensions with different scales.
    std = chains.reshape(-1, chains.shape[-1]).std(axis=0) + 1e-30
    M = chains.shape[0]
    w_pairs = []
    for i in range(M):
        for j in range(i + 1, M):
            for d in range(chains.shape[-1]):
                w_pairs.append(_wass1d(chains[i, :, d], chains[j, :, d]) / std[d])
    w_med = float(np.median(w_pairs)) if w_pairs else float("nan")
    w_max = float(np.max(w_pairs))    if w_pairs else float("nan")

    # Also per-seed marginal means — if seeds disagree by >0.5 std, flag.
    seed_means = chains.mean(axis=1)                           # (M, D)
    seed_mean_spread = (seed_means.std(axis=0) / std).max()
    return dict(
        n_seeds=int(chains.shape[0]),
        n_samples_per_seed=int(chains.shape[1]),
        dim=int(chains.shape[-1]),
        rhat=rhat.tolist(),
        rhat_max=float(rhat.max()),
        wass_normed_med=w_med,
        wass_normed_max=w_max,
        seed_mean_spread_over_std=float(seed_mean_spread),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="astro/results")
    ap.add_argument("--problem", default=None)
    ap.add_argument("--sampler", default=None)
    args = ap.parse_args()

    root = Path(args.results_dir)
    # Enumerate (problem, sampler) cells present in the tree.
    cells = sorted({(p.parts[-4], p.parts[-3])
                    for p in root.glob("*/*/seed*/samples.npz")})
    if args.problem: cells = [c for c in cells if c[0] == args.problem]
    if args.sampler: cells = [c for c in cells if c[1] == args.sampler]

    print(f"{'problem':<10s} {'sampler':<12s}  {'R̂ max':>8s}  "
          f"{'W (med)':>8s}  {'W (max)':>8s}  {'μ-spread':>10s}  status")
    print("-" * 78)
    for p, s in cells:
        r = analyze(p, s, root)
        if r is None:
            continue
        status = "ok"
        if r["rhat_max"] > 1.2:
            status = "BAD (trapped?)"
        elif r["rhat_max"] > 1.05:
            status = "warn"
        print(f"{p:<10s} {s:<12s}  {r['rhat_max']:8.3f}  "
              f"{r['wass_normed_med']:8.3f}  {r['wass_normed_max']:8.3f}  "
              f"{r['seed_mean_spread_over_std']:10.3f}  {status}")


if __name__ == "__main__":
    main()
