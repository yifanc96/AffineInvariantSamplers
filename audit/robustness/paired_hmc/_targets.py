"""Synthetic benchmark targets with analytic ground truth.

Two families, both with a batched log-density ``log_prob(theta)`` taking
shape ``(batch, D) -> (batch,)`` for ensemble samplers (the runner adapts
single-point APIs), an ``init`` helper, and an exact analytic
``truth`` dict (per-dimension mean and standard deviation) so that sampler
bias can be measured without a reference run.

1. Anisotropic Gaussian  N(0, Sigma)
   Eigen-standard-deviations have condition exp(log_cond); the covariance
   condition is exp(2 * log_cond) (one in dimension one).
   optionally rotated by a random orthogonal matrix so the covariance is
   not axis-aligned. Affine-invariant samplers are invariant to both the
   conditioning and the rotation; non-affine samplers degrade with kappa.

2. Paired Rosenbrock (even D)
   pi(q) ∝ exp( -sum_{i=1}^{D/2} [100 (q_{2i-1}^2 - q_{2i})^2 + (q_{2i-1}-1)^2] ).
   Banana-shaped curvature that no global affine map removes, so the
   gradient-free stretch move struggles while HMC-style dynamics traverse
   it. Each independent pair (a, b) has the closed-form marginals
       a ~ N(1, 1/2),   b | a ~ N(a^2, 1/200),
   giving E[a]=1, Var[a]=1/2, E[b]=3/2, Var[b]=1/200 + 5/2 = 2.505.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


# ─────────────────────────── Anisotropic Gaussian ──────────────────────────

def make_gaussian(dim: int, log_cond: float = 6.0, rotate: bool = True,
                  seed: int = 0):
    """N(0, Sigma); log_cond is the log standard-deviation condition.

    The covariance condition is exp(2 * log_cond), or one for dim=1.
    This preserves the historical target distribution for this argument.

    Returns (log_prob, truth, meta). ``truth`` has per-dim mean/std; for a
    rotated covariance the per-dim std is sqrt(diag(Sigma)).
    """
    rng = np.random.default_rng(seed)
    # log-spaced standard deviations: ratio max/min = exp(log_cond)
    log_sigma = np.linspace(-0.5, 0.5, dim) * log_cond
    sigma = np.exp(log_sigma)                      # eigen-stds
    var = sigma ** 2

    if rotate and dim > 1:
        # random orthogonal Q via QR of a Gaussian matrix
        Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    else:
        Q = np.eye(dim)

    Sigma = (Q * var) @ Q.T                        # Q diag(var) Q^T
    Sigma = 0.5 * (Sigma + Sigma.T)
    prec = np.linalg.inv(Sigma)
    diag_std = np.sqrt(np.diag(Sigma))

    prec_j = jnp.asarray(prec, dtype=jnp.float64)

    def log_prob(theta):                            # (B, D) -> (B,)
        return -0.5 * jnp.einsum("bi,ij,bj->b", theta, prec_j, theta)

    truth = dict(mean=np.zeros(dim), std=diag_std,
                 cov=Sigma, eigen_std=sigma)
    meta = dict(name="gaussian", dim=dim, log_cond=float(log_cond),
                log_cond_definition="log eigen-standard-deviation condition",
                kappa_std=float(sigma.max() / sigma.min()),
                kappa=float(var.max() / var.min()),
                kappa_cov=float(var.max() / var.min()), rotated=bool(rotate),
                param_names=[f"x{i}" for i in range(dim)])
    return log_prob, truth, meta


# ─────────────────────────── Paired Rosenbrock ─────────────────────────────

# closed-form per-pair moments
_ROSEN_PAIR_MEAN = np.array([1.0, 1.5])
_ROSEN_PAIR_STD = np.sqrt(np.array([0.5, 1.0 / 200.0 + 2.5]))   # [a, b]


def make_rosenbrock(dim: int):
    """Paired Rosenbrock; requires even ``dim``.

    Returns (log_prob, truth, meta) with analytic per-dim mean/std.
    """
    if dim % 2 != 0:
        raise ValueError(f"Rosenbrock requires even dim, got {dim}")

    def log_prob(theta):                            # (B, D) -> (B,)
        a = theta[:, 0::2]                          # odd components q_{2i-1}
        b = theta[:, 1::2]                          # even components q_{2i}
        return -jnp.sum(100.0 * (a ** 2 - b) ** 2 + (a - 1.0) ** 2, axis=1)

    n_pair = dim // 2
    mean = np.tile(_ROSEN_PAIR_MEAN, n_pair)
    std = np.tile(_ROSEN_PAIR_STD, n_pair)
    truth = dict(mean=mean, std=std)
    meta = dict(name="rosenbrock", dim=dim, n_pairs=n_pair,
                param_names=[f"q{i}" for i in range(dim)])
    return log_prob, truth, meta


# ─────────────────────────────── Init ──────────────────────────────────────

def init_ball(key, n_chains: int, dim: int, center=0.0, scale: float = 1.0):
    """Gaussian ball of walkers, shape (n_chains, dim)."""
    c = jnp.asarray(center) if np.ndim(center) else center
    return c + scale * jax.random.normal(key, (n_chains, dim))


def make_target(name: str, dim: int, **kw):
    """Dispatch by name: 'gaussian' or 'rosenbrock'."""
    if name == "gaussian":
        return make_gaussian(dim, **kw)
    if name == "rosenbrock":
        return make_rosenbrock(dim)
    raise ValueError(f"unknown target {name!r}")
