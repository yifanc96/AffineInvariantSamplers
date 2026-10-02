"""Statistically rigorous accuracy + mixing metrics.

Samples have shape (N, n_chains, D). We use ArviZ's rank-normalized,
split-chain estimators (Vehtari et al. 2021), which are the community
standard and handle multi-chain ensembles and finite-sample fluctuation
correctly:

- ESS (bulk): effective sample size per dimension. For antithetic chains
  (well-tuned NUTS on a Gaussian) ESS can exceed N·C, which is correct.
- MCSE: Monte-Carlo standard error of the mean, = posterior_sd / sqrt(ESS).
- IAT: derived from the rigorous ESS as τ = N·C / ESS (worst dim = bottleneck).

Burn-in / systematic bias is measured against the *analytic* truth and
judged relative to MC fluctuation via a z-score:

    bias_z = |mean_hat − mean_true| / MCSE

Under a converged, unbiased sampler mean_hat ~ Normal(mean_true, MCSE²), so
bias_z ~ O(1). bias_z ≫ 1 on some coordinate is statistically significant
evidence of residual burn-in / systematic bias beyond Monte-Carlo noise.
We report the max over dimensions (the worst coordinate).
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
    N, C, D = samples.shape
    s2 = samples.reshape(N * C, D).var(0)                  # pooled marginal var
    m = samples.mean(1)                                    # (N, D) ensemble mean
    var_m = m.var(0)
    tau = np.array([_iat_1d(m[:, j]) for j in range(D)])
    infl = np.where(var_m > 0, s2 / np.maximum(var_m, 1e-300), C)  # ~C if independent
    ess = infl * (N / np.maximum(tau, 1.0))
    return ess, tau


def sd_from_ess(samples, ess):
    """MCSE of the mean per coordinate from a given ESS: sd_pooled / sqrt(ESS)."""
    N, C, D = samples.shape
    sd = samples.reshape(N * C, D).std(0)
    return sd / np.sqrt(np.maximum(ess, 1e-30))


def rigorous_metrics(samples, truth, n_grad_evals=None, fast=False):
    """Mixing + accuracy metrics.

    fast=True replaces the per-chain rank-normalized ArviZ ESS (which does one
    FFT per chain per coordinate and blows up at large walker counts) with the
    O(D N log N) ensemble-mean ESS. rhat is kept (needed for convergence gating)
    but computed on a capped subset of chains so it stays cheap. Use fast=True
    for large ensembles (e.g. the high-d scaling sweep)."""
    N, C, D = samples.shape

    if fast:
        ess_bulk, tau_ens = ensemble_ess(samples)
        # split-half stationarity from the ensemble mean (cheap)
        half = N // 2
        e1, _ = ensemble_ess(samples[:half]); e2, _ = ensemble_ess(samples[half:])
        split_ratio = e1 / np.maximum(e2, 1e-30)
        # rhat on a capped subset of chains (rank-normalized split-Rhat is O(C N log N))
        cap = min(C, 256)
        idx = np.linspace(0, C - 1, cap).astype(int)
        rhat = np.asarray(az.rhat(_chain_draw(samples[:, idx, :])),
                          dtype=float).reshape(-1)
        mcse_mean = np.asarray(sd_from_ess(samples, ess_bulk), dtype=float)
    else:
        cd = _chain_draw(samples)
        ess_bulk = np.asarray(az.ess(cd), dtype=float).reshape(-1)       # (D,)
        mcse_mean = np.asarray(az.mcse(cd), dtype=float).reshape(-1)     # (D,)
        rhat = np.asarray(az.rhat(cd), dtype=float).reshape(-1)          # (D,)
        # split-half stationarity: ESS of 1st vs 2nd half should agree if
        # stationary; ratio far from 1 flags drift
        half = N // 2
        e1 = np.asarray(az.ess(cd[:, :half, :]), dtype=float).reshape(-1)
        e2 = np.asarray(az.ess(cd[:, half:, :]), dtype=float).reshape(-1)
        split_ratio = e1 / np.maximum(e2, 1e-30)
        ess_ens, _ = ensemble_ess(samples)                              # cross-check field

    flat = samples.reshape(N * C, D)
    mu_hat, sd_hat = flat.mean(0), flat.std(0)
    mu_true = np.asarray(truth["mean"], dtype=float)
    sd_true = np.asarray(truth["std"], dtype=float)

    mean_abs = np.abs(mu_hat - mu_true)
    mean_sigma = mean_abs / np.maximum(sd_true, 1e-12)               # bias in posterior σ
    bias_z = mean_abs / np.maximum(mcse_mean, 1e-30)                 # bias vs MC fluctuation
    std_rel = np.abs(sd_hat - sd_true) / np.maximum(sd_true, 1e-12)
    iat = (N * C) / np.maximum(ess_bulk, 1e-30)                      # effective IAT per dim

    out = dict(
        ess_bulk_min=float(np.nanmin(ess_bulk)),
        ess_bulk_median=float(np.nanmedian(ess_bulk)),
        iat_max=float(np.nanmax(iat)),
        iat_median=float(np.nanmedian(iat)),
        mean_err_sigma_max=float(mean_sigma.max()),
        mean_err_sigma_mean=float(mean_sigma.mean()),
        std_rel_err_max=float(std_rel.max()),
        bias_z_max=float(np.nanmax(bias_z)),
        bias_z_median=float(np.nanmedian(bias_z)),
        rhat_max=float(np.nanmax(rhat)),               # ~1.00 if mixed/stationary
        split_ess_ratio_med=float(np.nanmedian(split_ratio)),  # ~1 if stationary
        ess_method=("ensemble" if fast else "arviz_rank"),
    )
    if not fast:
        out["ess_ens_min"] = float(np.nanmin(ess_ens))    # cheap cross-check field
    if n_grad_evals:
        out["ess_per_grad"] = float(np.nanmin(ess_bulk) / n_grad_evals)
        if not fast:
            out["ess_per_grad_ens"] = float(np.nanmin(ess_ens) / n_grad_evals)
    return out


# back-compat alias used by run.py
def all_metrics(samples, truth, n_grad_evals=None, fast=False):
    return rigorous_metrics(samples, truth, n_grad_evals, fast=fast)
