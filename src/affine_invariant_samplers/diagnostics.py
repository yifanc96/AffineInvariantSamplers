"""
affine_invariant_samplers.diagnostics
=====================================

MCMC diagnostics: autocorrelation function, integrated autocorrelation time,
and effective sample size.

All functions accept samples of shape (N,), (N, D), or (N, n_chains, D):
  * (N,)            — one chain, one parameter
  * (N, D)          — one chain, D parameters
  * (N, n_chains, D) — full output of sampler_xxx(...); chains are averaged

The integrated autocorrelation time follows Sokal's self-consistent-window
estimator (used also by emcee). ESS = N_total / τ.

Per-walker averages omit correlations between different walkers at different
times. Inspect ensemble-mean observables and independent ensemble runs as well.
Constant series have undefined autocorrelation; tau=inf and ESS=0 are used as
conservative reporting sentinels rather than evidence of independent samples.
"""

from __future__ import annotations

import numpy as np
from operator import index


__all__ = [
    "autocorrelation_fft",
    "integrated_autocorr_time_1d",
    "autocorrelation",
    "integrated_autocorr_time",
    "effective_sample_size",
]


# ──────────────────────────────────────────────────────────────────────────────
# 1D primitives (port of initial-samplers/autocorrelation_func.py)
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_max_lag(n, max_lag):
    """Choose a positive ACF length that fits the available samples."""
    if n < 2:
        raise ValueError("autocorrelation requires at least 2 samples")
    if max_lag is None:
        return max(1, min(n // 3, 20_000))
    try:
        max_lag = index(max_lag)
    except TypeError as exc:
        raise ValueError("max_lag must be a positive integer") from exc
    if max_lag < 1:
        raise ValueError("max_lag must be a positive integer")
    return min(max_lag, n)


def autocorrelation_fft(x, max_lag=None):
    """FFT-based autocorrelation function for a 1D sequence.

    Args:
        x        : 1D array-like of samples.
        max_lag  : Number of lags to return, including lag zero. Defaults to
                   max(1, min(N // 3, 20000)) and is capped at N.

    Returns:
        acf : 1D numpy array of length min(max_lag, N). acf[0] == 1.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 2:
        raise ValueError("autocorrelation_fft requires at least 2 samples")
    if not np.isfinite(x).all():
        raise ValueError("autocorrelation samples must all be finite")
    max_lag = _normalize_max_lag(n, max_lag)

    x_norm = x - x.mean()
    var = x_norm.var()
    if var <= 0.0:
        # Constant series: ACF is technically undefined; return lag-0 = 1, else 0.
        out = np.zeros(max_lag)
        out[0] = 1.0
        return out
    x_norm /= np.sqrt(var)

    # Zero-pad to avoid circular correlation
    fft = np.fft.fft(x_norm, n=2 * n)
    acf = np.fft.ifft(fft * np.conjugate(fft))[:n].real / n
    return acf[:max_lag]


def integrated_autocorr_time_1d(x, M=5, c=10, max_iterations=10):
    """Integrated autocorrelation time τ for a 1D sequence.

    Uses Sokal's self-consistent-window rule: τ = 1 + 2·Σ_{k=1}^{W} ρ(k),
    with W the smallest integer such that W ≥ M·τ(W).  If the chain is too
    short for a robust estimate, pairwise-averages (halving) the series and
    rescales τ ← 2^k · τ_k.

    Args:
        x              : 1D array-like.
        M              : Window multiplier, default 5.
        c              : Robustness cutoff.  Estimate is accepted when
                         N ≥ c · τ.
        max_iterations : Cap on pairwise-averaging passes.

    Returns:
        tau : float, integrated autocorrelation time.
        acf : numpy array, autocorrelation at original time scale.
        ess : float, effective sample size = N_original / τ.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 2:
        raise ValueError("integrated_autocorr_time_1d requires at least 2 samples")
    if not np.isfinite(x).all():
        raise ValueError("autocorrelation samples must all be finite")
    if M <= 0 or c <= 0:
        raise ValueError("M and c must be positive")
    try:
        max_iterations = index(max_iterations)
    except TypeError as exc:
        raise ValueError("max_iterations must be a nonnegative integer") from exc
    if max_iterations < 0:
        raise ValueError("max_iterations must be a nonnegative integer")

    if np.all(x == x[0]):
        return float("inf"), autocorrelation_fft(x), 0.0

    orig_x = x.copy()
    n_orig = orig_x.size
    n = n_orig
    tau = 1.0

    k = 0
    converged = False
    while k < max_iterations:
        if x.size < 2:
            break
        acf = autocorrelation_fft(x)
        tau = 1.0
        for window in range(1, len(acf)):
            tau_w = 1.0 + 2.0 * float(acf[1:window + 1].sum())
            if window <= M * tau_w:
                tau = tau_w
            else:
                break
        if n >= c * tau:
            tau = tau * (2 ** k)
            converged = True
            break
        # pairwise reduction
        k += 1
        n_half = n // 2
        x_new = np.zeros(n_half)
        for i in range(n_half):
            j = 2 * i
            x_new[i] = 0.5 * (x[j] + x[j + 1]) if j + 1 < n else x[j]
        x = x_new
        n = x.size

    # final recompute at original scale
    acf = autocorrelation_fft(orig_x)
    if not converged:
        # This ACF is already on the original time scale: multiplying by
        # the reduction factor again would inflate tau with each retry.
        tau = 1.0 + 2.0 * float(acf[1: min(len(acf), int(M) + 1)].sum())

    return float(tau), acf, float(n_orig / tau)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-dimensional wrappers over sampler output (N, n_chains, D)
# ──────────────────────────────────────────────────────────────────────────────

def _to_3d(samples):
    """Normalize `(N,)`, `(N, D)`, `(N, n_chains, D)` → `(N, n_chains, D)`."""
    arr = np.asarray(samples, dtype=float)
    if arr.ndim == 1:
        return arr[:, None, None]          # (N, 1, 1)
    if arr.ndim == 2:
        # (N, D) — interpret as one chain with D dims
        return arr[:, None, :]             # (N, 1, D)
    if arr.ndim == 3:
        return arr                         # (N, n_chains, D)
    raise ValueError(f"samples must have ndim 1, 2, or 3, got {arr.ndim}")


def autocorrelation(samples, max_lag=None):
    """Autocorrelation function per dimension, averaged across chains.

    Args:
        samples : (N,) or (N, D) or (N, n_chains, D) array.
        max_lag : Number of lags including zero, capped at N. Default:
                  max(1, min(N // 3, 20_000)).

    Returns:
        acf : array of shape `(min(max_lag, N), D)`. `acf[0] == 1`.
    """
    s = _to_3d(samples)
    N, C, D = s.shape
    max_lag = _normalize_max_lag(N, max_lag)
    out = np.zeros((max_lag, D))
    for d in range(D):
        acc = np.zeros(max_lag)
        for c in range(C):
            acc += autocorrelation_fft(s[:, c, d], max_lag=max_lag)
        out[:, d] = acc / C
    return out


def integrated_autocorr_time(samples, M=5, c=10):
    """Integrated autocorrelation time per dimension, averaged across chains.

    Args:
        samples : (N,) or (N, D) or (N, n_chains, D).
        M, c    : Passed to the self-consistent-window estimator.

    Returns:
        tau : 1D array of length D.  Scalar if input was 1D.
    """
    s = _to_3d(samples)
    N, C, D = s.shape
    taus = np.zeros((C, D))
    for c_i in range(C):
        for d in range(D):
            taus[c_i, d], _, _ = integrated_autocorr_time_1d(s[:, c_i, d], M=M, c=c)
    tau = taus.mean(axis=0)   # average across chains per dim
    if np.asarray(samples).ndim == 1:
        return float(tau[0])
    return tau


def effective_sample_size(samples, M=5, c=10):
    """Effective sample size per dimension, capped at the sample count.

    Args:
        samples : (N,) or (N, D) or (N, n_chains, D).
        M, c    : Passed to the self-consistent-window estimator.

    Returns:
        ess : 1D array of length D, giving `(N · n_chains) / max(τ_d, 1)`.
              Scalar if input was 1D.

    Notes:
        Well-tuned HMC can produce antithetic samples (ρ(1) < 0), for which
        a window estimator can return τ < 1 and a naive ESS = N / τ would
        exceed N.  As a reporting policy, we clamp τ at 1 inside this
        function so ESS is bounded above by the actual sample count.
        `integrated_autocorr_time` does *not* clamp — its τ < 1 is useful
        diagnostic information.
        This implementation can miss strong alternating correlations; compare
        additional observables and independent ensemble runs.
    """
    s = _to_3d(samples)
    N, C, D = s.shape
    tau = np.asarray(integrated_autocorr_time(samples, M=M, c=c))
    tau_eff = np.maximum(tau, 1.0)
    n_total = N * C
    ess = n_total / tau_eff
    if np.asarray(samples).ndim == 1:
        return float(ess)
    return ess
