"""Extra benchmarks where HMC is *supposed* to beat stretch/emcee:
low-D curved geometry (Neal's funnel) and high-D ill-conditioned Gaussian
scaling.  Not astrophysics posteriors — these are canonical HMC tests
that expose affine-invariant ensemble samplers' known weaknesses.

Neal's funnel
-------------
    v ~ N(0, 3)
    x_i ~ N(0, exp(v))  for i = 1..D-1
Curvature spans e^3 ≈ 20× across v.  Stretch struggles in the neck because
one walker pair can't span both the wide and narrow regions simultaneously;
HMC with (affine-invariant) preconditioning adapts.

Ill-conditioned Gaussian
------------------------
    x ~ N(0, Σ),  Σ = diag(σ_1^2, …, σ_D^2),  σ_i = exp(s·(i/D − 0.5))
Eigenvalues span e^{s} (default s=6 → 400× spread).  Stretch ESS
degrades as D × cond(Σ); HMC with a preconditioner is insensitive to
condition number.  Varying D tests *dimensional scaling* directly.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


# ─── Neal's funnel ───────────────────────────────────────────────────────────

def make_funnel(dim: int = 20):
    """Neal's funnel in `dim` dimensions.  θ = (v, x_1, …, x_{D-1}).

    Uniform prior box [-30, 30] on v, [-1e4, 1e4] on x to keep finite support
    (actual posterior decays fast enough these are effectively non-binding).
    """
    assert dim >= 2
    LO = jnp.concatenate([jnp.array([-30.0]), jnp.full(dim - 1, -1e4)])
    HI = jnp.concatenate([jnp.array([ 30.0]), jnp.full(dim - 1,  1e4)])

    def log_prob(theta):
        v    = theta[:, 0]
        x    = theta[:, 1:]
        in_box = jnp.all((theta >= LO) & (theta <= HI), axis=1)
        log_pv = -0.5 * (v / 3.0) ** 2 - jnp.log(3.0) - 0.5 * jnp.log(2 * jnp.pi)
        # N(x_i | 0, exp(v/2)^2) but commonly stated as N(0, exp(v)) — use the
        # Neal formulation: σ^2 = exp(v) → log σ = v/2.
        log_px = (-0.5 * jnp.sum(x ** 2, axis=1) * jnp.exp(-v)
                  - 0.5 * (dim - 1) * v
                  - 0.5 * (dim - 1) * jnp.log(2 * jnp.pi))
        lp = log_pv + log_px
        return jnp.where(in_box, lp, -jnp.inf)

    info = dict(dim=dim, n_data=0,
                param_names=["v"] + [f"x{i}" for i in range(1, dim)],
                prior_lo=np.asarray(LO), prior_hi=np.asarray(HI))
    return log_prob, info


def init_funnel(key, n_chains: int, dim: int = 20):
    """Small ball around the mean of the prior (v=0, x=0)."""
    v0 = 0.3 * jax.random.normal(key, (n_chains, 1))
    x0 = 1.0 * jax.random.normal(jax.random.fold_in(key, 1), (n_chains, dim - 1))
    return jnp.concatenate([v0, x0], axis=1)


# ─── Ill-conditioned Gaussian ────────────────────────────────────────────────

def make_illcond_gauss(dim: int = 50, log_cond: float = 6.0):
    """Diagonal Gaussian with condition number exp(log_cond).

    σ_i = exp(log_cond · (i/D − 0.5)), so the std ratio max/min = exp(log_cond).
    Default log_cond=6 ⇒ 403× spread; log_cond=10 ⇒ 22000×.
    """
    idx = jnp.arange(dim, dtype=jnp.float32)
    log_sigma = log_cond * (idx / dim - 0.5)
    sigma_i   = jnp.exp(log_sigma)
    inv_var   = 1.0 / (sigma_i ** 2)

    LO = jnp.full(dim, -1e4)
    HI = jnp.full(dim,  1e4)

    def log_prob(theta):
        in_box = jnp.all((theta >= LO) & (theta <= HI), axis=1)
        lp = -0.5 * jnp.sum(theta ** 2 * inv_var[None, :], axis=1)
        return jnp.where(in_box, lp, -jnp.inf)

    info = dict(dim=dim, n_data=0, log_cond=log_cond,
                sigma=np.asarray(sigma_i),
                param_names=[f"x{i}" for i in range(dim)],
                prior_lo=np.asarray(LO), prior_hi=np.asarray(HI))
    return log_prob, info


def init_illcond_gauss(key, n_chains: int, dim: int = 50):
    """Small ball around the origin — deliberately smaller than the posterior
    in *every* direction to test warmup.  Each HMC warmup + stretch pre-warmup
    must discover the per-dim scale."""
    return 0.1 * jax.random.normal(key, (n_chains, dim))
