"""DES Y1 3x2pt — sampled in PHYSICAL parameter space (no logit reparam).

This is the same forward model as ``likelihoods_des_y1`` but parameters are
sampled on their physical scales with wide Gaussian priors instead of the
non-centered logit/Normal-decentered z-space transform. The point: physical
parameters span 5 orders of magnitude in marginal scale (σ_dz3=0.011 vs
σ_A=2.89), giving κ(Σ) ≈ 9M. Tests whether affine-invariant ensembles can
automate the preconditioning that the cosmology community does manually.

Priors:
  - Uniform-prior parameters → wide Gaussian centered on (lo+hi)/2,
    std = (hi-lo)/4 (≈95% mass inside the original Uniform box).
  - Normal-prior parameters → unchanged.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

from astro.likelihoods_des_y1 import (
    _UNIF, _NORM, _UNIF_LO, _UNIF_HI, _NORM_MU, _NORM_SD,
    _FID, _DIM, _theory_and_data,
)

# Wide-Gaussian replacement for the Uniform prior:
# center on midpoint, std = range/4
_UNIF_MU = 0.5 * (_UNIF_LO + _UNIF_HI)
_UNIF_SD = (_UNIF_HI - _UNIF_LO) / 4.0

# Stack into full physical-space prior mean/std
_PHYS_MU = jnp.concatenate([_UNIF_MU, _NORM_MU])
_PHYS_SD = jnp.concatenate([_UNIF_SD, _NORM_SD])


def make_des_y1_phys():
    theory_fn, data_mean, cov, fid_phys = _theory_and_data()

    cov_np = np.asarray(cov, dtype=np.float64)
    cov_np = 0.5 * (cov_np + cov_np.T) + 1e-10 * np.eye(cov_np.shape[0])
    L = np.linalg.cholesky(cov_np)
    L_j = jnp.asarray(L)
    data_mean_j = jnp.asarray(data_mean)

    def _one_lp(theta):
        cl = theory_fn(theta)
        r  = data_mean_j - cl
        u  = jax.scipy.linalg.solve_triangular(L_j, r, lower=True)
        log_lik = -0.5 * jnp.sum(u * u)
        # Wide Gaussian prior in physical space (no Jacobian needed).
        log_prior = -0.5 * jnp.sum(((theta - _PHYS_MU) / _PHYS_SD) ** 2)
        return log_lik + log_prior

    def log_prob(theta):
        if theta.ndim == 1:
            return _one_lp(theta)
        return jax.vmap(_one_lp)(theta)

    def residual_fn(theta):
        cl = theory_fn(theta)
        r_lik   = jax.scipy.linalg.solve_triangular(L_j, data_mean_j - cl, lower=True)
        r_prior = (theta - _PHYS_MU) / _PHYS_SD
        return jnp.concatenate([r_lik, r_prior])

    param_names = [n for n, _, _ in _UNIF] + [n for n, _, _ in _NORM]
    info = dict(dim=_DIM, n_data=int(data_mean.size),
                param_names=param_names,
                prior_lo=np.asarray(_PHYS_MU - 5 * _PHYS_SD),
                prior_hi=np.asarray(_PHYS_MU + 5 * _PHYS_SD),
                phys_fid=np.asarray(fid_phys),
                residual_fn=residual_fn)
    return log_prob, info


def init_des_y1_phys(key, n_chains: int, init_scale: float = 0.1):
    """Walkers spawn around fiducial point in physical space.

    Width of the ball is ``init_scale * _PHYS_SD`` per dim — so init_scale=1.0
    is a one-prior-std ball, init_scale=0.1 is tight-around-fiducial.
    """
    fid_phys = jnp.asarray(
        [_FID["Omega_c"], _FID["sigma8"], _FID["Omega_b"],
         _FID["h"], _FID["n_s"], _FID["w0"],
         _FID["A"], _FID["eta"],
         *_FID["b"], *_FID["m"], *_FID["dz"]])
    return fid_phys[None, :] + init_scale * _PHYS_SD * jax.random.normal(
        key, (n_chains, _DIM))
