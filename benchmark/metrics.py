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


def rigorous_metrics(samples, truth, n_grad_evals=None):
    N, C, D = samples.shape
    cd = _chain_draw(samples)

    ess_bulk = np.asarray(az.ess(cd), dtype=float).reshape(-1)       # (D,)
    mcse_mean = np.asarray(az.mcse(cd), dtype=float).reshape(-1)     # (D,)
    rhat = np.asarray(az.rhat(cd), dtype=float).reshape(-1)          # (D,)

    # split-half stationarity: ESS of 1st vs 2nd half should agree (~0.5 each
    # of the full ESS) if the chain is stationary; ratio far from 1 flags drift
    half = N // 2
    e1 = np.asarray(az.ess(cd[:, :half, :]), dtype=float).reshape(-1)
    e2 = np.asarray(az.ess(cd[:, half:, :]), dtype=float).reshape(-1)
    split_ratio = e1 / np.maximum(e2, 1e-30)

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
    )
    if n_grad_evals:
        out["ess_per_grad"] = float(np.nanmin(ess_bulk) / n_grad_evals)
    return out


# back-compat alias used by run.py
def all_metrics(samples, truth, n_grad_evals=None):
    return rigorous_metrics(samples, truth, n_grad_evals)
