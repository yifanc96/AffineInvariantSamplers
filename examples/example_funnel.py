"""
example_funnel.py — Neal's funnel benchmark.

Target  (D = 1 + d, here d = D - 1):
    v ~ N(0, 9)
    x_i | v ~ N(0, exp(v))    i = 1, ..., d

so  log pi(v, x) = -v^2/18 - d·v/2 - 0.5·exp(-v)·||x||^2 + const.

The tight neck at v ≪ 0 makes this a classical stress-test for MCMC.

Exact moments:
    E[v] = 0, Var[v] = 9
    E[x_i] = 0, Var[x_i] = E[exp(v)] = exp(9/2) ≈ 90.02

Samplers compared:
    * sampler_stretch                — Goodman–Weare stretch
    * sampler_ensemble_dr_stretch    — 2-stage delayed-rejection stretch
    * sampler_gndr                   — Gauss–Newton proposal Langevin + DR
"""

from __future__ import annotations

import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from affine_invariant_samplers import (
    sampler_stretch,
    sampler_ensemble_dr_stretch,
    sampler_gndr,
    effective_sample_size,
)
from affine_invariant_samplers.plotting import corner_plot


# ──────────────────────────────────────────────────────────────────────────────
# Target:  Neal's funnel
# ──────────────────────────────────────────────────────────────────────────────

def make_funnel(dim=5):
    """Build log_prob (batched), log_prob_single, residual_fn (for gndr)."""
    d = dim - 1   # number of x-dimensions

    def log_prob(x):                                   # (batch, D) -> (batch,)
        v  = x[:, 0]
        xs = x[:, 1:]
        log_p_v = -0.5 * v ** 2 / 9.0
        log_p_x = (-0.5 * jnp.sum(xs ** 2 * jnp.exp(-v)[:, None], axis=1)
                   - 0.5 * d * v)
        return log_p_v + log_p_x

    def log_prob_single(z):                            # (D,) -> scalar
        v  = z[0]
        xs = z[1:]
        return (-0.5 * v ** 2 / 9.0
                - 0.5 * d * v
                - 0.5 * jnp.sum(xs ** 2) * jnp.exp(-v))

    def residual(z):                                   # (D,) -> (D,)
        # r = [v/3, exp(-v/2) * x_1, ..., exp(-v/2) * x_d]
        # ||r||^2 = v^2/9 + exp(-v)*||x||^2  => H_GN = J^T J captures
        # the v-dependent scaling of the x-components.
        v  = z[0]
        xs = z[1:]
        return jnp.concatenate([jnp.array([v / 3.0]),
                                jnp.exp(-v / 2.0) * xs])

    return log_prob, log_prob_single, residual


# ──────────────────────────────────────────────────────────────────────────────
# Report helper
# ──────────────────────────────────────────────────────────────────────────────

def _report(name, samples, accept, elapsed):
    flat = np.asarray(samples.reshape(-1, samples.shape[-1]))
    v    = flat[:, 0]
    xs   = flat[:, 1:]
    mv   = float(np.mean(v))
    vv   = float(np.var(v))
    vxs  = float(np.mean(np.var(xs, axis=0)))
    ess  = effective_sample_size(samples)
    ess_v = float(ess[0])
    print(f"  {name:<22s}  v: mean={mv:6.3f} var={vv:6.3f}   "
          f"x_i var={vxs:7.2f}   accept={accept:5.3f}   "
          f"ESS(v)={ess_v:6.1f}   time={elapsed:5.1f}s")


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dim      = 5
    n_chains = 50
    n_samp   = 20_000
    warmup   = 4_000
    seed     = 1

    log_prob, log_prob_single, residual = make_funnel(dim=dim)
    init_ens = jax.random.normal(jax.random.key(99), (n_chains, dim))
    init_hmc = jax.random.normal(jax.random.key(99), (n_chains, dim))   # fewer for gndr (DR is expensive)

    print(f"Neal's funnel  D={dim}  n_chains={n_chains}  n_samp={n_samp}  warmup={warmup}")
    print("=" * 100)

    results = {}

    t0 = time.time()
    s, info = sampler_stretch(log_prob, init_ens, n_samp, warmup=warmup,
                               seed=seed, verbose=False)
    _report("stretch",         s, info["acceptance_rate"], time.time() - t0)
    results["stretch"] = s

    t0 = time.time()
    s, info = sampler_ensemble_dr_stretch(log_prob, init_ens, n_samp, warmup=warmup,
                                           seed=seed, shrink=0.3, verbose=False)
    _report("stretch-DR",      s, info["acceptance_rate"], time.time() - t0)
    results["stretch-DR"] = s

    t0 = time.time()
    s, info = sampler_gndr(log_prob_single, init_hmc, n_samp, warmup=warmup,
                            step_size=0.5, n_try=3, residual_fn=residual,
                            seed=seed, shrink = 0.3, verbose=False)
    _report("gndr",            s, info["acceptance_rate"], time.time() - t0)
    results["gndr"] = s

    print("=" * 100)
    print("Target:  v mean=0, var=9   x_i var = exp(9/2) ≈ 90.02")

    # ──────────────────────────────────────────────────────────────────────
    # Plots
    # ──────────────────────────────────────────────────────────────────────

    # Funnel 2D marginal of (v, x_0).  x_i are iid given v, so
    #   p(v, x_0) ∝ exp(-v²/18) · exp(-v/2) · exp(-x_0² · exp(-v) / 2)
    v_grid = np.linspace(-9, 6, 400)
    x_grid = np.linspace(-15, 15, 400)
    V, X   = np.meshgrid(v_grid, x_grid)
    true_density = (np.exp(-0.5 * V ** 2 / 9.0)
                    * np.exp(-0.5 * V)                # one x-dim prefactor
                    * np.exp(-0.5 * X ** 2 * np.exp(-V)))
    true_density /= true_density.max()

    fig_c, axes_c = plt.subplots(1, len(results), figsize=(4.3 * len(results), 4.2),
                                  sharex=True, sharey=True)
    for ax, (name, s) in zip(axes_c, results.items()):
        flat = np.asarray(s).reshape(-1, dim)
        # subsample for clarity — too many points obscure the funnel shape
        rng = np.random.default_rng(0)
        idx = rng.choice(flat.shape[0], size=min(20_000, flat.shape[0]),
                         replace=False)
        ax.scatter(flat[idx, 0], flat[idx, 1], s=2, alpha=0.15, color="C0")
        ax.contour(V, X, true_density, levels=8, colors="k",
                   linewidths=0.8, alpha=0.7)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("v")
        ax.set_xlim(v_grid[0], v_grid[-1])
        ax.set_ylim(x_grid[0], x_grid[-1])
    axes_c[0].set_ylabel("x₀")
    fig_c.suptitle("Neal's funnel 2D marginal (v, x₀)  (black = true contours)",
                   y=0.99)
    fig_c.tight_layout()

    # Per-method corner plots over all D = 5 dims
    labels = ["v"] + [f"x{i}" for i in range(dim - 1)]
    truths = [0.0] * dim
    for name, s in results.items():
        fig = corner_plot(np.asarray(s).reshape(-1, dim),
                          labels=labels, truths=truths,
                          title=f"{name} — funnel")

    plt.show()
