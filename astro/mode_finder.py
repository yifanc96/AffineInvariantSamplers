"""MAP / mode finder for HMC sampler initialisation.

Standard practice in pymc / numpyro / bilby: before running an HMC-family
sampler on an unfamiliar posterior, run a quick L-BFGS or Adam on
``-log_prob`` to locate a rough MAP, then spawn walkers in a small ball
around it.  Sampler step-size adaptation then sees local curvature near
the typical set rather than wherever the initial ensemble happened to land.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def find_mode(log_prob, x0, n_steps: int = 300, lr: float = 0.05,
              bounds=None):
    """Adam on a single-point ``-log_prob``.  Returns (x_map, lp_map).

    Uses Adam (no optax dep).  For each of ``n_steps``:
        g = ∇ (-log_prob)(x)
        x ← adam_step(x, g), then project into `bounds` if given

    `x0` may be (D,) for a single start, or (B, D) for parallel restarts
    (returns the best of B).

    `bounds` : optional (lo, hi) tuple of 1-D arrays — if given, the optimiser
    projects into the box after each step AND reverts any step that caused
    ``log_prob`` to drop to -inf (which happens when Adam leaps across a
    hard prior boundary; without this guard all walkers would anchor at a
    -inf point where the gradient is 0 and the rest of the pipeline silently
    fails).  Critical for problems with hard prior boxes — transit, gw_pop.
    """
    x0 = jnp.asarray(x0)
    single = x0.ndim == 1
    if single:
        x0 = x0[None, :]

    lo = hi = None
    if bounds is not None:
        lo = jnp.asarray(bounds[0])
        hi = jnp.asarray(bounds[1])

    def neg_lp(x):
        return -log_prob(x[None, :])[0]

    grad_fn = jax.jit(jax.vmap(jax.grad(neg_lp)))
    lp_fn   = jax.jit(log_prob)

    m = jnp.zeros_like(x0)
    v = jnp.zeros_like(x0)
    b1, b2, eps = 0.9, 0.999, 1e-8
    x = x0
    if bounds is not None:
        x = jnp.clip(x, lo, hi)
    lp_prev = lp_fn(x)

    for t in range(1, n_steps + 1):
        g = grad_fn(x)
        g = jnp.where(jnp.isfinite(g), g, 0.0)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        mh = m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        x_new = x - lr * mh / (jnp.sqrt(vh) + eps)
        if bounds is not None:
            x_new = jnp.clip(x_new, lo, hi)

        # Revert any walker whose log-prob collapsed to -inf (stepped across a
        # soft boundary inside the box, or hit a numerical degeneracy).
        lp_new = lp_fn(x_new)
        accept = jnp.isfinite(lp_new) & (lp_new >= lp_prev - 50.0)
        x = jnp.where(accept[:, None], x_new, x)
        lp_prev = jnp.where(accept, lp_new, lp_prev)

    lp = log_prob(x)
    best = int(jnp.argmax(lp))
    x_map = x[best]
    lp_map = float(lp[best])
    if single:
        return np.asarray(x_map), lp_map
    return np.asarray(x), np.asarray(lp)


def ball_around_mode(log_prob, x0, n_chains: int,
                      scale: float = 1e-2, seed: int = 0,
                      n_steps: int = 300, lr: float = 0.05,
                      bounds=None, ensemble_scale=None):
    """Find MAP, then draw ``n_chains`` walkers from a Gaussian ball around it.

    Parameters
    ----------
    ensemble_scale : optional (D,) vector
        Per-coordinate ball width.  If given, overrides the heuristic
        ``scale * (|MAP| + 1)``.  Typically set to the std of the user's
        initial ensemble — this preserves the spread the user chose,
        re-centered on MAP.  Critical: too-tight a ball (e.g. the 0.01·|MAP|
        default) makes the posterior look locally Gaussian to dual-averaging
        step-size adaptation, which then collapses ε to zero chasing a
        target-acceptance it cannot reach.

    If ``bounds=(lo, hi)`` is given, the mode-finder stays inside the box
    and the ball is re-clipped.
    """
    x_map, lp_map = find_mode(log_prob, x0, n_steps=n_steps, lr=lr, bounds=bounds)
    if ensemble_scale is not None:
        sigma = jnp.asarray(ensemble_scale)
    else:
        sigma = scale * (jnp.abs(jnp.asarray(x_map)) + 1.0)
    ball  = x_map[None, :] + sigma[None, :] * jax.random.normal(
        jax.random.key(seed), (n_chains, x_map.shape[0]))
    if bounds is not None:
        lo, hi = jnp.asarray(bounds[0]), jnp.asarray(bounds[1])
        ball = jnp.clip(ball, lo, hi)
    return ball, dict(x_map=x_map.tolist(), lp_map=lp_map,
                       sigma=np.asarray(sigma).tolist())
