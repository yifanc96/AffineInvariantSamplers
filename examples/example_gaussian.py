"""
example_gaussian.py — High-dimensional anisotropic Gaussian benchmark.

Target:
    log pi(x) = -0.5 x^T Prec x
    where Prec = Q · diag(1/lambda) · Q^T, with `lambda` log-spaced from 1 to
    `kappa` (condition number) and Q a random orthogonal basis.

Samplers compared:
    * sampler_stretch       — Goodman–Weare stretch (affine-invariant, MH)
    * sampler_langevin_walk — Langevin walk (ensemble-preconditioned MALA)
    * sampler_kalman_move   — ensemble Kalman move (derivative-free drift)
    * sampler_peaches       — ensemble-preconditioned HMC (walk + HMC)

For each, we report mean relative variance error and minimum ESS across
dimensions.
"""

from __future__ import annotations

import time
import jax
import jax.numpy as jnp

from affine_invariant_samplers import (
    sampler_stretch,
    sampler_langevin_walk,
    sampler_kalman_move,
    sampler_peaches,
    effective_sample_size,
)


# ──────────────────────────────────────────────────────────────────────────────
# Target:  ill-conditioned Gaussian
# ──────────────────────────────────────────────────────────────────────────────

def make_gaussian(dim=20, kappa=1000.0, seed=0):
    """Return (log_prob_batched, cov, prec)."""
    eigvals = jnp.logspace(0, jnp.log10(kappa), dim)
    Q, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(seed), (dim, dim)))
    cov  = Q @ jnp.diag(eigvals) @ Q.T
    prec = Q @ jnp.diag(1.0 / eigvals) @ Q.T

    def log_prob(x):                          # (batch, D) -> (batch,)
        return -0.5 * jnp.sum((x @ prec) * x, axis=-1)

    return log_prob, cov, prec


# ──────────────────────────────────────────────────────────────────────────────
# Report helper
# ──────────────────────────────────────────────────────────────────────────────

def _report(name, samples, cov, accept, elapsed):
    flat = jnp.asarray(samples).reshape(-1, samples.shape[-1])
    var_est  = jnp.var(flat, axis=0)
    var_true = jnp.diag(cov)
    rel_err  = float(jnp.mean(jnp.abs(var_est - var_true) / var_true))
    ess      = effective_sample_size(samples)
    print(f"  {name:<20s}  rel_err(var)={rel_err:6.3f}   "
          f"accept={accept:5.3f}   min_ESS={float(ess.min()):7.1f}   "
          f"time={elapsed:5.1f}s")


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dim      = 20
    kappa    = 1000.0
    n_chains = 100
    n_samp   = 5000
    warmup   = 1000
    seed     = 123

    log_prob, cov, prec = make_gaussian(dim=dim, kappa=kappa, seed=0)
    init = jax.random.normal(jax.random.key(42), (n_chains, dim))

    print(f"Anisotropic Gaussian  D={dim}  kappa={kappa}  "
          f"n_chains={n_chains}  n_samp={n_samp}  warmup={warmup}")
    print("=" * 80)

    t0 = time.time()
    s, info = sampler_stretch(log_prob, init, n_samp, warmup=warmup, seed=seed,
                              verbose=False)
    _report("stretch",       s, cov, info["acceptance_rate"], time.time() - t0)

    t0 = time.time()
    s, info = sampler_langevin_walk(log_prob, init, n_samp, warmup=warmup,
                                     step_size=0.01, seed=seed, verbose=False)
    _report("langevin_walk", s, cov, info["acceptance_rate"], time.time() - t0)

    t0 = time.time()
    s, info = sampler_kalman_move(log_prob, lambda x: x, prec, init, n_samp,
                                   warmup=warmup, step_size=0.01, seed=seed,
                                   verbose=False)
    _report("kalman_move",   s, cov, info["acceptance_rate"], time.time() - t0)

    t0 = time.time()
    s, info = sampler_peaches(log_prob, init, n_samp, warmup=warmup,
                               step_size=0.01, seed=seed, verbose=False)
    _report("peaches",       s, cov, info["acceptance_rate"], time.time() - t0)

    print("=" * 80)
    print(f"Target: diag(cov) spans [{float(jnp.diag(cov).min()):.2f}, "
          f"{float(jnp.diag(cov).max()):.2f}]  (ratio = kappa = {kappa})")
