"""Accuracy and mixing diagnostics for samples shaped (draw, walker, dim).

ArviZ rank ESS/R-hat assume independent chains; walkers in one interacting
ensemble do not satisfy that assumption. They are descriptive cross-checks,
not a convergence certificate. The ensemble-mean estimator includes cross-
walker time dependence, but still needs a sufficiently long stationary run.
Its Sokal window clamps IAT to at least one, so it differs from rank ESS for
antithetic trajectories. Compare independent ensemble seeds and analytic
first/second moments alongside both explicitly labeled ESS estimates.
"""
from __future__ import annotations

import numpy as np
import arviz as az


def _chain_draw(samples):
    # (N, C, D) -> (C, N, D): ArviZ wants (chain, draw, dim)
    return np.ascontiguousarray(np.transpose(samples, (1, 0, 2)))


# ---------------------------------------------------------------------------
# Fast ensemble ESS (avoids the per-chain FFT blow-up of az.ess at large C).
#
# For an ensemble of W walkers each sampling pi, the estimator of E[f] is the
# grand mean over walkers AND iterations. Its variance is governed by the
# ensemble-MEAN time series m_t = (1/W) sum_w f(x_{w,t}): a SINGLE length-N
# series per coordinate. With tau_m its integrated autocorrelation time and
# s2 the pooled marginal variance,
#     Var(mean) = Var(m) * tau_m / N,   ESS = s2 / Var(mean) * ... = (s2/Var(m)) * N/tau_m.
# For independent walkers s2/Var(m) = W, recovering ESS = W*N/tau_m; the ratio
# corrects for equal-time cross-walker correlation. Cost is O(D * N log N),
# independent of W -- so 1024 walkers is as cheap to score as 8.
# ---------------------------------------------------------------------------
def _iat_1d(x, c=5.0):
    """Integrated autocorrelation time of a 1-D series (Sokal auto-window)."""
    n = x.shape[0]
    x = x - x.mean()
    if n < 4:
        return 1.0
    nfft = 1 << (2 * n - 1).bit_length()          # next pow2 >= 2n-1
    f = np.fft.rfft(x, n=nfft)
    acf = np.fft.irfft(f * np.conj(f), n=nfft)[:n].real
    if acf[0] <= 0:
        return 1.0
    acf /= acf[0]
    taus = 2.0 * np.cumsum(acf) - 1.0             # running estimate 1 + 2 sum_{k<=m} rho_k
    win = np.arange(n) < c * taus                 # Sokal window: m >= c*tau(m)
    m = int(np.argmin(win)) if not win.all() else n - 1
    return float(max(taus[m], 1.0))


def ensemble_ess(samples):
    """Worst-coordinate ensemble ESS from the ensemble-mean autocorrelation.

    samples: (N, C, D). Returns (ess_per_dim (D,), tau_m_per_dim (D,))."""
    samples = np.asarray(samples, dtype=float)
    N, C, D = samples.shape
    if N < 4:
        return np.zeros(D), np.full(D, np.inf)
    s2 = samples.reshape(N * C, D).var(0)
    m = samples.mean(1)
    var_m = m.var(0)
    # Frozen walkers can have the exact target marginal variance. They provide
    # no temporal information; a constant ensemble mean has no estimable IAT.
    moving = np.any(samples[1:] != samples[:-1], axis=(0, 1))
    valid = (np.all(np.isfinite(samples), axis=(0, 1)) & moving &
             (s2 > 0) & (var_m > 0))
    tau = np.full(D, np.inf)
    ess = np.zeros(D)
    for j in np.flatnonzero(valid):
        tau[j] = _iat_1d(m[:, j])
        ess[j] = (s2[j] / var_m[j]) * N / tau[j]
    return ess, tau


def sd_from_ess(samples, ess):
    """MCSE; an unestimable/zero ESS has infinite uncertainty."""
    N, C, D = samples.shape
    sd = samples.reshape(N * C, D).std(0)
    out = np.full(D, np.inf)
    np.divide(sd, np.sqrt(ess), out=out, where=np.asarray(ess) > 0)
    return out


def _finite_stat(values, op):
    values = np.asarray(values)
    return float(op(values)) if np.all(np.isfinite(values)) else float("nan")


def rigorous_metrics(samples, truth, n_grad_evals=None, fast=False):
    """Mixing + accuracy metrics.

    fast=True replaces the per-chain rank-normalized ArviZ ESS (which does one
    FFT per chain per coordinate and blows up at large walker counts) with the
    O(D N log N) ensemble-mean ESS. Descriptive R-hat is computed on a capped
    subset of walkers so it stays cheap. Use fast=True
    for large ensembles (e.g. the high-d scaling sweep)."""
    samples = np.asarray(samples, dtype=float)
    N, C, D = samples.shape
    ess_ens, tau_ens = ensemble_ess(samples)
    ess_squared, _ = ensemble_ess(samples ** 2)
    invalid = ess_ens == 0
    if fast:
        ess_bulk = ess_ens.copy()
        half = N // 2
        e1, _ = ensemble_ess(samples[:half])
        e2, _ = ensemble_ess(samples[half:])
        rhat_samples = samples[:, np.linspace(0, C - 1, min(C, 256)).astype(int), :]
        mcse_mean = sd_from_ess(samples, ess_bulk)
    else:
        cd = _chain_draw(samples)
        ess_bulk = np.asarray(az.ess(cd), dtype=float).reshape(-1)
        mcse_mean = np.asarray(az.mcse(cd), dtype=float).reshape(-1)
        half = N // 2
        e1 = np.asarray(az.ess(cd[:, :half, :]), dtype=float).reshape(-1)
        e2 = np.asarray(az.ess(cd[:, half:, :]), dtype=float).reshape(-1)
        rhat_samples = samples
    rhat = (np.asarray(az.rhat(_chain_draw(rhat_samples)), dtype=float).reshape(-1)
            if N >= 4 and C >= 2 else np.full(D, np.nan))
    # Do not let rank normalization or coincidentally correct initial moments
    # award a frozen ensemble positive ESS or a finite uncertainty estimate.
    ess_bulk[invalid] = 0
    mcse_mean[invalid] = np.inf
    rhat[invalid] = np.inf
    split_ratio = np.full(D, np.nan)
    np.divide(e1, e2, out=split_ratio, where=(e1 > 0) & (e2 > 0) & ~invalid)

    flat = samples.reshape(N * C, D)
    mu_hat, sd_hat = flat.mean(0), flat.std(0)
    mu_true = np.asarray(truth["mean"], dtype=float)
    sd_true = np.asarray(truth["std"], dtype=float)

    mean_abs = np.abs(mu_hat - mu_true)
    mean_sigma = mean_abs / np.maximum(sd_true, 1e-12)               # bias in posterior σ
    bias_z = mean_abs / np.maximum(mcse_mean, 1e-30)                 # bias vs MC fluctuation
    std_rel = np.abs(sd_hat - sd_true) / np.maximum(sd_true, 1e-12)
    bias_z[invalid] = np.nan
    iat = np.full(D, np.inf)
    np.divide(N * C, ess_bulk, out=iat, where=ess_bulk > 0)
    second_true = sd_true ** 2 + mu_true ** 2
    second_rel = np.abs((flat ** 2).mean(0) - second_true) / np.maximum(second_true, 1e-12)

    out = dict(
        ess_bulk_min=float(np.nanmin(ess_bulk)),
        ess_bulk_median=float(np.nanmedian(ess_bulk)),
        iat_max=float(np.nanmax(iat)),
        iat_median=float(np.nanmedian(iat)),
        mean_err_sigma_max=float(mean_sigma.max()),
        mean_err_sigma_mean=float(mean_sigma.mean()),
        std_rel_err_max=float(std_rel.max()),
        bias_z_max=_finite_stat(bias_z, np.max),
        bias_z_median=_finite_stat(bias_z, np.median),
        rhat_max=float(np.nanmax(rhat)),               # ~1.00 if mixed/stationary
        split_ess_ratio_med=_finite_stat(split_ratio, np.median),  # ~1 if stationary
        ess_method=("ensemble" if fast else "arviz_rank"),
        ess_ensemble_method="ensemble_mean_sokal_iat_clamped_at_one",
        ess_ensemble_mean_min=float(np.min(ess_ens)),
        ess_ensemble_second_moment_min=float(np.min(ess_squared)),
        second_moment_rel_err_max=float(np.max(second_rel)),
        unestimable_mean_dimensions=np.flatnonzero(invalid).tolist(),
        unestimable_second_moment_dimensions=np.flatnonzero(ess_squared == 0).tolist(),
        diagnostic_caveat="Within-ensemble walkers are dependent; rank ESS/R-hat do not certify convergence.",
    )
    if not fast:
        out["ess_rank_bulk_min"] = float(np.min(ess_bulk))
        out["ess_ens_min"] = float(np.nanmin(ess_ens))    # cheap cross-check field
    if n_grad_evals:
        out["ess_per_grad"] = float(np.nanmin(ess_bulk) / n_grad_evals)
        if not fast:
            out["ess_per_grad_ens"] = float(np.nanmin(ess_ens) / n_grad_evals)
    return out


# back-compat alias used by run.py
def all_metrics(samples, truth, n_grad_evals=None, fast=False):
    return rigorous_metrics(samples, truth, n_grad_evals, fast=fast)
