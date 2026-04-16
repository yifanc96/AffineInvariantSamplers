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

def _report(name, samples, info, elapsed):
    flat = np.asarray(samples.reshape(-1, samples.shape[-1]))
    v    = flat[:, 0]
    xs   = flat[:, 1:]
    mv   = float(np.mean(v))
    vv   = float(np.var(v))
    vxs  = float(np.mean(np.var(xs, axis=0)))
    ess  = effective_sample_size(samples)
    ess_v = float(ess[0])
    accept = info["acceptance_rate"]
    grads  = info.get("n_grad_evals")
    grads_s = f"{grads:>9d}" if grads is not None else f"{'–':>9s}"
    print(f"  {name:<22s}  v: mean={mv:6.3f} var={vv:6.3f}   "
          f"x_i var={vxs:7.2f}   accept={accept:5.3f}   "
          f"ESS(v)={ess_v:6.1f}   grad_evals={grads_s}   "
          f"time={elapsed:5.1f}s")


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
    _report("stretch",         s, info, time.time() - t0)
    results["stretch"] = s

    t0 = time.time()
    s, info = sampler_ensemble_dr_stretch(log_prob, init_ens, n_samp, warmup=warmup,
                                           seed=seed, shrink=0.3, verbose=False)
    _report("stretch-DR",      s, info, time.time() - t0)
    results["stretch-DR"] = s

    t0 = time.time()
    s, info = sampler_gndr(log_prob_single, init_hmc, n_samp, warmup=warmup,
                            step_size=0.5, n_try=3, residual_fn=residual,
                            seed=seed, shrink = 0.3, verbose=False)
    _report("gndr",            s, info, time.time() - t0)
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

    # --- 1D marginal of v (should be N(0, 9)) ---
    v_lin = np.linspace(-12, 8, 400)
    v_true_pdf = np.exp(-0.5 * v_lin ** 2 / 9.0) / np.sqrt(2 * np.pi * 9.0)
    fig_v, axes_v = plt.subplots(1, len(results),
                                  figsize=(4.3 * len(results), 3.2),
                                  sharey=True)
    for ax, (name, s) in zip(axes_v, results.items()):
        v = np.asarray(s).reshape(-1, dim)[:, 0]
        ax.hist(v, bins=80, range=(v_lin[0], v_lin[-1]), density=True,
                color="C0", alpha=0.55, edgecolor="C0", histtype="stepfilled",
                label="samples")
        ax.plot(v_lin, v_true_pdf, "k-", linewidth=1.3, label="N(0, 9)")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("v")
        ax.set_xlim(v_lin[0], v_lin[-1])
    axes_v[0].set_ylabel("density")
    axes_v[-1].legend(fontsize=8, loc="upper right", frameon=False)
    fig_v.suptitle("Marginal of v  (true: N(0, 9))", y=0.99)
    fig_v.tight_layout()

    # --- 2D marginal of (v, x_0) ---
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

    # Per-method corner plots over all D = 5 dims, with analytical marginals.
    labels = ["v"] + [f"x{i}" for i in range(dim - 1)]
    truths = [0.0] * dim

    # 1D marginals:
    #   v ~ N(0, 9)
    #   x_i is marginal ∫ N(x_i; 0, e^v) · N(v; 0, 9) dv — evaluate numerically.
    v_grid_1d = np.linspace(-12, 8, 400)
    truth_1d_f = {0: (v_grid_1d, np.exp(-0.5 * v_grid_1d ** 2 / 9.0)
                                  / np.sqrt(2 * np.pi * 9.0))}
    vv = np.linspace(-9, 6, 300)
    pv = np.exp(-0.5 * vv ** 2 / 9.0) / np.sqrt(2 * np.pi * 9.0)
    x_grid_1d = np.linspace(-15, 15, 400)
    # p(x_i) = ∫ N(x_i; 0, e^v) p(v) dv  — trapezoidal quadrature
    pdf_x = np.zeros_like(x_grid_1d)
    for vk, pk in zip(vv, pv):
        sig2 = np.exp(vk)
        pdf_x += pk * np.exp(-0.5 * x_grid_1d ** 2 / sig2) / np.sqrt(2 * np.pi * sig2)
    pdf_x *= (vv[1] - vv[0])
    for i in range(1, dim):
        truth_1d_f[i] = (x_grid_1d, pdf_x)

    # 2D marginals for (x_i, v) pairs (i = 1..d).  p(v, x_i) = p(v) · N(x_i; 0, e^v).
    V, X = np.meshgrid(v_grid_1d, x_grid_1d)
    pvx = (np.exp(-0.5 * V ** 2 / 9.0) / np.sqrt(2 * np.pi * 9.0)) * \
          (np.exp(-0.5 * X ** 2 * np.exp(-V)) / np.sqrt(2 * np.pi * np.exp(V)))
    truth_2d_f = {}
    for i in range(1, dim):
        truth_2d_f[(i, 0)] = (v_grid_1d, x_grid_1d, pvx)

    for name, s in results.items():
        fig = corner_plot(np.asarray(s).reshape(-1, dim),
                          labels=labels, truths=truths,
                          truth_1d=truth_1d_f, truth_2d=truth_2d_f,
                          title=f"{name} — funnel")

    plt.show()
