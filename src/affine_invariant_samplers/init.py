"""
affine_invariant_samplers.init
==============================

Initialization utilities for the samplers in this package.

The MCMC samplers here take an *ensemble* ``init`` of shape ``(n_chains, D)``.
A good starting ensemble is roughly concentrated on the bulk of the target and
dispersed on the right scale — if it is under-dispersed (e.g. ``N(0, I)`` when
the target has variance ~100), the ``find_init_step_size`` heuristic latches
onto the tiny ensemble covariance and picks a step size that is too large.

This module provides three tiers of helpers:

    Tier 1 — ``find_map``
        Single-start BFGS minimizer of ``-log_prob_single``.  Cheap, fine for
        smooth unimodal targets.

    Tier 2 — ``find_map_restarts``
        Multi-start BFGS via ``vmap``.  Returns the best minimum found.
        Handles mild multimodality and poor initial guesses.

    Tier 3 — ``init_ensemble_from_map``
        Finds the MAP, computes the Hessian there (Laplace approximation),
        and draws ``n_chains`` samples ``x_map + eps @ L^T`` with
        ``L L^T = H^{-1}``.  Gives the sampler a well-conditioned ensemble
        matched to the local curvature.  Costs one Hessian — expensive in D,
        but you only pay it once.

Caveats
-------
* **Tier 3 can be catastrophically wrong even when the MAP exists.**
  Neal's funnel has MAP at ``(v*, x*) = (-9d/2, 0)`` and BFGS finds it in
  a handful of steps — but at that point the Hessian for ``x`` is
  ``exp(-v*)·I`` (~e¹⁸ in 5-D), so Laplace predicts ``std(x_i) ≈ 10⁻³``
  against a true marginal std ``≈ 9.5``.  Use Tiers 1 or 2 on funnel-like
  targets and seed your own dispersed ensemble around the MAP point.
* **Multimodal targets.**  MAP is a point estimate — it seeds all chains
  in one mode.  Use ``find_map_restarts`` with many restarts and inspect
  the spread, or fall back to a dispersed random init.
* **Non-smooth / unbounded potentials.**  BFGS needs twice-differentiability.
  Targets with genuinely unbounded potentials (no MAP at all) will send
  the optimizer to infinity.
* The Hessian at an early-stopped optimum may be indefinite; we regularize
  by clipping eigenvalues to a positive floor.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.optimize


__all__ = [
    "find_map",
    "find_map_restarts",
    "init_ensemble_from_map",
]


# ──────────────────────────────────────────────────────────────────────────────
# Tier 1: single-start BFGS
# ──────────────────────────────────────────────────────────────────────────────

class MAPResult(NamedTuple):
    """Result of a MAP finder.

    Attributes
    ----------
    x : jnp.ndarray, shape (D,)
        Minimizer of ``-log_prob_single``.
    fun : float
        Value of ``-log_prob_single`` at ``x``.
    success : bool
        Whether the optimizer reported convergence.
    nit : int
        Number of iterations taken.
    """
    x: jnp.ndarray
    fun: float
    success: bool
    nit: int


def find_map(log_prob_single: Callable,
             x0: jnp.ndarray,
             maxiter: int = 500,
             verbose: bool = False) -> MAPResult:
    """Find the MAP of a target density by BFGS on the negative log-prob.

    Parameters
    ----------
    log_prob_single : callable
        ``(D,) -> scalar`` — log-density of the target at a single point.
    x0 : array, shape (D,)
        Starting point.
    maxiter : int
        Maximum BFGS iterations.
    verbose : bool
        Print a one-line status on exit.

    Returns
    -------
    MAPResult
        ``(x, fun, success, nit)``.
    """
    x0 = jnp.asarray(x0, dtype=jnp.float64 if jax.config.jax_enable_x64 else jnp.float32)
    neg = lambda x: -log_prob_single(x)
    res = jax.scipy.optimize.minimize(
        neg, x0, method="BFGS", options={"maxiter": maxiter},
    )
    out = MAPResult(x=res.x, fun=float(res.fun),
                    success=bool(res.success), nit=int(res.nit))
    if verbose:
        status = "converged" if out.success else "did NOT converge"
        print(f"[find_map] BFGS {status} in {out.nit} steps, "
              f"-log_prob = {out.fun:.4g}, ||x_MAP|| = {float(jnp.linalg.norm(out.x)):.4g}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Tier 2: multi-start BFGS via vmap
# ──────────────────────────────────────────────────────────────────────────────

def find_map_restarts(log_prob_single: Callable,
                      x0_batch: jnp.ndarray,
                      maxiter: int = 500,
                      verbose: bool = False) -> MAPResult:
    """Run BFGS from a batch of starting points and return the best minimum.

    Uses ``vmap`` to run all restarts in parallel on the accelerator.  Good
    when a single start is unreliable — either because the landscape has
    several basins, or because you don't trust your initial guess.

    Parameters
    ----------
    log_prob_single : callable
        ``(D,) -> scalar``.
    x0_batch : array, shape (n_restarts, D)
        Starting points.  Usually something like
        ``sigma * jax.random.normal(key, (n_restarts, D))``.
    maxiter : int
        Max BFGS iterations per restart.
    verbose : bool
        Print summary of restarts.

    Returns
    -------
    MAPResult
        Best minimum across restarts.
    """
    x0_batch = jnp.asarray(x0_batch)
    neg = lambda x: -log_prob_single(x)

    def _one(x0):
        res = jax.scipy.optimize.minimize(
            neg, x0, method="BFGS", options={"maxiter": maxiter},
        )
        return res.x, res.fun, res.success, res.nit

    xs, funs, succ, nits = jax.vmap(_one)(x0_batch)
    i_best = int(jnp.argmin(funs))
    out = MAPResult(x=xs[i_best], fun=float(funs[i_best]),
                    success=bool(succ[i_best]), nit=int(nits[i_best]))
    if verbose:
        n_ok = int(jnp.sum(succ))
        n_total = x0_batch.shape[0]
        spread = float(jnp.max(funs) - jnp.min(funs))
        print(f"[find_map_restarts] {n_ok}/{n_total} restarts converged, "
              f"best -log_prob = {out.fun:.4g}, spread across restarts = {spread:.4g}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Tier 3: MAP + Laplace → ensemble
# ──────────────────────────────────────────────────────────────────────────────

def init_ensemble_from_map(log_prob_single: Callable,
                           x0: jnp.ndarray,
                           n_chains: int,
                           seed: int = 0,
                           n_restarts: int = 1,
                           scale: float = 1.0,
                           eig_floor: float = 1e-6,
                           maxiter: int = 500,
                           verbose: bool = False):
    """Return ``(init, map_result)`` — a Laplace-seeded ensemble.

    Computes the MAP, the Hessian of ``-log_prob_single`` there, regularizes
    it to be positive-definite, and draws ``n_chains`` samples from the
    Gaussian ``N(x_MAP, scale^2 · H^{-1})``.  This is the Laplace
    approximation to the target; for mildly non-Gaussian targets it gives a
    far better starting ensemble than isotropic noise.

    Parameters
    ----------
    log_prob_single : callable
        ``(D,) -> scalar``.  Used both for optimization and for the Hessian,
        so it must be twice-differentiable under JAX.
    x0 : array, shape (D,) or (n_restarts, D)
        Starting guess for the optimizer.  If 2-D and ``n_restarts > 1``,
        each row is used as a restart; otherwise ``x0`` is tiled.
    n_chains : int
        Number of chains in the returned ensemble.
    seed : int
        PRNG seed for the Laplace draw.
    n_restarts : int
        Number of BFGS restarts.  If > 1, uses ``find_map_restarts``.
    scale : float
        Multiplier on the Laplace covariance.  ``1.0`` matches the local
        curvature; ``<1`` gives a tighter init, ``>1`` a more dispersed one.
    eig_floor : float
        Hessian eigenvalues below this are clipped up (regularization for
        indefinite / ill-conditioned Hessians at early-stopped optima).
    maxiter : int
        Max BFGS iterations.
    verbose : bool
        Print MAP + Laplace summary.

    Returns
    -------
    init : jnp.ndarray, shape (n_chains, D)
        Ensemble drawn from the Laplace approximation at the MAP.
    map_result : MAPResult
        Diagnostics from the optimizer.
    """
    x0 = jnp.asarray(x0)

    # ── MAP ───────────────────────────────────────────────────────────────
    if n_restarts > 1:
        if x0.ndim == 1:
            x0_batch = jnp.broadcast_to(x0, (n_restarts, x0.shape[0]))
            # add a bit of noise so restarts don't all collapse to the same path
            key = jax.random.key(seed)
            x0_batch = x0_batch + 0.1 * jax.random.normal(key, x0_batch.shape)
        else:
            x0_batch = x0
        res = find_map_restarts(log_prob_single, x0_batch,
                                 maxiter=maxiter, verbose=verbose)
    else:
        x0_vec = x0 if x0.ndim == 1 else x0[0]
        res = find_map(log_prob_single, x0_vec,
                        maxiter=maxiter, verbose=verbose)

    x_map = res.x
    D = x_map.shape[0]

    # ── Hessian + regularized inverse cholesky ────────────────────────────
    H = jax.hessian(lambda x: -log_prob_single(x))(x_map)
    # Symmetrize (autodiff can produce tiny asymmetry)
    H = 0.5 * (H + H.T)
    # Eigendecomp, clip eigenvalues, reconstruct inverse
    eigvals, eigvecs = jnp.linalg.eigh(H)
    eigvals_clipped = jnp.maximum(eigvals, eig_floor)
    inv_sqrt_eigvals = 1.0 / jnp.sqrt(eigvals_clipped)
    # Draws:  x_map + scale * eigvecs @ diag(1/sqrt(eig)) @ z,  z ~ N(0, I)
    key = jax.random.key(seed + 1)
    z = jax.random.normal(key, (n_chains, D))
    # (scale * eigvecs @ diag(1/sqrt(eig)))  is the Cholesky-like factor L
    # such that  L L^T = H^{-1} (up to regularization)
    L = scale * eigvecs * inv_sqrt_eigvals[None, :]     # (D, D)
    init = x_map[None, :] + z @ L.T                     # (n_chains, D)

    if verbose:
        cond = float(eigvals_clipped.max() / eigvals_clipped.min())
        n_clipped = int(jnp.sum(eigvals < eig_floor))
        print(f"[init_ensemble_from_map] D={D}, Laplace cov condition number = {cond:.3g}"
              + (f", {n_clipped} eigenvalue(s) clipped at floor {eig_floor:g}"
                 if n_clipped else ""))
        init_std = float(jnp.std(init, axis=0).mean())
        print(f"   drew {n_chains} chains, mean per-dim std = {init_std:.3g}")

    return init, res
