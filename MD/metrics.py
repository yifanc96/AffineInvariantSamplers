"""Permutation/rotation-invariant evaluation metrics for MD benchmarks.

- ``w2_1d``: exact 1-D Wasserstein-2 via sorting.
- ``energy_w2``: W2 between two energy-value samples.
- ``distance_w2``: W2 between two pooled pairwise-distance samples
  (symmetries: particle permutations and global rotations/translations
  are all absorbed by looking only at the multiset of pairwise distances).
"""
from __future__ import annotations

import numpy as np


def w2_1d(a: np.ndarray, b: np.ndarray, max_n: int = 200_000) -> float:
    """Exact 1-D Wasserstein-2 between two empirical samples.

    For 1-D data the optimal transport is obtained by sorting both
    samples.  Unequal sizes are handled by resampling to ``n =
    max(|a|, |b|)`` via evenly-spaced quantile interpolation, but we
    cap ``n`` at ``max_n`` to avoid multi-gigabyte sorts on LJ-55-scale
    distance pools.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")

    n = min(max(a.size, b.size), max_n)

    if a.size == b.size and a.size <= max_n:
        return float(np.sqrt(np.mean((np.sort(a) - np.sort(b)) ** 2)))

    qs = (np.arange(n) + 0.5) / n

    def q(x):
        s = np.sort(x)
        if s.size == n:
            return s
        return np.interp(qs, (np.arange(s.size) + 0.5) / s.size, s)

    return float(np.sqrt(np.mean((q(a) - q(b)) ** 2)))


def pairwise_distances_flat(samples: np.ndarray, n_particles: int,
                            spatial_dim: int, batch: int = 4096) -> np.ndarray:
    """Return all pairwise distances pooled across samples.

    samples: (N, n_particles * spatial_dim)
    Returns a 1-D array of length N * n_particles*(n_particles-1)/2.

    Batched over N to avoid (N, n, n) temporaries on large LJ-55 arrays.
    """
    iu, ju = np.triu_indices(n_particles, k=1)
    parts = []
    N = samples.shape[0]
    for i in range(0, N, batch):
        coords = samples[i:i + batch].reshape(-1, n_particles, spatial_dim)
        diffs = coords[:, iu, :] - coords[:, ju, :]
        parts.append(np.sqrt(np.sum(diffs * diffs, axis=-1)).ravel())
    return np.concatenate(parts)


def energy_w2(energies_a: np.ndarray, energies_b: np.ndarray) -> float:
    """W2 of 1-D energy distributions."""
    return w2_1d(energies_a, energies_b)


def distance_w2(samples_a: np.ndarray, samples_b: np.ndarray,
                n_particles: int, spatial_dim: int) -> float:
    """Pairwise-distance W2: permutation- and rotation-invariant proxy for
    the geometric Wasserstein-2 used by the Adjoint Sampling paper.
    """
    da = pairwise_distances_flat(samples_a, n_particles, spatial_dim)
    db = pairwise_distances_flat(samples_b, n_particles, spatial_dim)
    return w2_1d(da, db)
