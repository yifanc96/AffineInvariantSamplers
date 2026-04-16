"""
example_rosenbrock_unadjusted.py — Unadjusted Langevin methods on 10-D Rosenbrock.

Compares two unadjusted (no Metropolis correction) samplers on Rosenbrock:
    * sampler_aldi                — affine-invariant Langevin dynamics
    * sampler_pickles_unadjusted  — kinetic Langevin with ensemble preconditioning

Both target the continuous-time invariant distribution.  Finite step size
introduces an O(h²) discretisation bias — larger h ⇒ faster mixing, more bias.

Both methods are ensemble-preconditioned, but differ in their dynamics:

  * `sampler_aldi` is **overdamped** Langevin (first-order in position).
    The ensemble covariance provides the affine-invariant preconditioner,
    but without momentum the sampler has to follow the curved banana point-
    by-point.  It cannot "glide" along the narrow valley the way an
    inertial method can — expect visible bias in both mean and variance on
    Rosenbrock.

  * `sampler_pickles_unadjusted` uses **kinetic (underdamped)** Langevin
    with the same ensemble preconditioning but a BAOAB momentum update.
    The momentum lets trajectories traverse the curved valley in long
    strokes; combined with PGN step-size rescaling this tracks Rosenbrock
    much better and recovers the target moments reasonably well.

Same target as `example_rosenbrock.py`:  (a, b) = (1, 100), D = 10.
Exact marginals:  x_even ~ N(a, 1/2)        mean = 1, var = 0.5
                  x_odd  | x_even ~ N(x_even², 1/(2b))
                                             mean = 1.5, var ≈ 2.505.
"""

from __future__ import annotations

import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from affine_invariant_samplers import (
    sampler_aldi,
    sampler_pickles_unadjusted,
    effective_sample_size,
)
from affine_invariant_samplers.plotting import corner_plot


# ──────────────────────────────────────────────────────────────────────────────
# Target:  10-D Rosenbrock
# ──────────────────────────────────────────────────────────────────────────────

def make_rosenbrock(dim=10, a=1.0, b=100.0):
    assert dim % 2 == 0

    def log_prob(x):                          # (batch, D) -> (batch,)
        xe = x[:, ::2]
        xo = x[:, 1::2]
        return -(b * jnp.sum((xo - xe ** 2) ** 2, axis=1)
                 + jnp.sum((xe - a) ** 2, axis=1))

    return log_prob


# ──────────────────────────────────────────────────────────────────────────────
# Report helper
# ──────────────────────────────────────────────────────────────────────────────

def _report(name, samples, a, info, elapsed):
    flat    = jnp.asarray(samples).reshape(-1, samples.shape[-1])
    xe, xo  = flat[:, ::2], flat[:, 1::2]
    me, ve  = float(jnp.mean(xe)), float(jnp.mean(jnp.var(xe, axis=0)))
    mo, vo  = float(jnp.mean(xo)), float(jnp.mean(jnp.var(xo, axis=0)))
    ess     = effective_sample_size(samples)
    grads   = info.get("n_grad_evals")
    grads_s = f"{grads:>10d}" if grads is not None else f"{'–':>10s}"
    print(f"  {name:<24s}  x_e mean={me:5.2f} var={ve:5.2f}   "
          f"x_o mean={mo:5.2f} var={vo:5.2f}   "
          f"min_ESS={float(ess.min()):7.1f}   grad_evals={grads_s}   "
          f"time={elapsed:5.1f}s")


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dim      = 10
    a, b     = 1.0, 100.0
    n_chains = 100
    n_samp   = 10000
    warmup   = 2000
    seed     = 123

    log_prob = make_rosenbrock(dim=dim, a=a, b=b)
    init = jax.random.normal(jax.random.key(42), (n_chains, dim))

    print(f"Unadjusted Langevin on Rosenbrock  D={dim}  (a, b) = ({a}, {b})  "
          f"n_chains={n_chains}  n_samp={n_samp}  warmup={warmup}")
    print("=" * 110)

    results = {}

    # aldi: step_size=0.1 is a reasonable compromise.  Too small and x_even
    # barely migrates off the origin; too large and x_odd becomes unstable.
    # Even so, aldi is expected to show visible bias on Rosenbrock.
    t0 = time.time()
    s, info = sampler_aldi(log_prob, init, n_samp, warmup=warmup,
                            step_size=0.1, seed=seed, verbose=False)
    _report("aldi",               s, a, info, time.time() - t0)
    results["aldi"] = s

    # pickles_unadjusted: its PGN adaptation rescales h by the ensemble
    # gradient norm, so a base step of 0.2 is safe and much larger than
    # what aldi can tolerate.  Expect nearly-correct moments.
    t0 = time.time()
    s, info = sampler_pickles_unadjusted(
        log_prob, init, n_samp, warmup=warmup,
        step_size=0.2, gamma=2.0, seed=seed, verbose=False)
    _report("pickles_unadjusted", s, a, info, time.time() - t0)
    results["pickles_unadjusted"] = s

    print("=" * 110)
    print(f"Target: x_e mean=1.00, var=0.50   x_o mean=1.50, var≈2.505")

    # ──────────────────────────────────────────────────────────────────────
    # Plots
    # ──────────────────────────────────────────────────────────────────────

    # 2D marginal of (x_0, x_1) with exact contours
    xr = np.linspace(-2.5, 2.5, 400)
    yr = np.linspace(-1.5, 6.0, 400)
    gx, gy = np.meshgrid(xr, yr)
    true_density = np.exp(-(b * (gy - gx ** 2) ** 2 + (gx - a) ** 2))

    fig_c, axes_c = plt.subplots(1, len(results), figsize=(4.4 * len(results), 4.2),
                                  sharex=True, sharey=True)
    for ax, (name, s) in zip(axes_c, results.items()):
        flat = np.asarray(s).reshape(-1, dim)
        ax.hist2d(flat[:, 0], flat[:, 1], bins=80,
                  range=[[xr[0], xr[-1]], [yr[0], yr[-1]]],
                  cmap="Blues", density=True)
        ax.contour(gx, gy, true_density, levels=8, colors="k",
                   linewidths=0.8, alpha=0.7)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("x₀ (even)")
    axes_c[0].set_ylabel("x₁ (odd)")
    fig_c.suptitle("Unadjusted Langevin, Rosenbrock (x₀, x₁) marginal  "
                   "(black = true contours)",
                   y=0.99)
    fig_c.tight_layout()

    # Per-method corner plots on the first 4 dims with full truth overlays
    # (see example_rosenbrock.py for the derivation).
    K = 4
    labels = [f"x{i}" + (" (e)" if i % 2 == 0 else " (o)") for i in range(K)]
    truths = [a if i % 2 == 0 else a ** 2 + 0.5 for i in range(K)]

    xe_grid = np.linspace(-2.0, 4.0, 300)
    xe_pdf  = np.exp(-(xe_grid - a) ** 2) / np.sqrt(np.pi)

    xo_grid = np.linspace(-1.0, 6.0, 300)
    xe_q = np.linspace(-3.0, 5.0, 600)
    p_xe = np.exp(-(xe_q - a) ** 2) / np.sqrt(np.pi)
    dxe  = xe_q[1] - xe_q[0]
    sqrt_b_over_pi = np.sqrt(b / np.pi)
    xo_pdf = np.zeros_like(xo_grid)
    for xe_k, pk in zip(xe_q, p_xe):
        xo_pdf += pk * sqrt_b_over_pi * np.exp(-b * (xo_grid - xe_k ** 2) ** 2)
    xo_pdf *= dxe

    def grid_and_pdf_1d(i):
        return (xe_grid, xe_pdf) if i % 2 == 0 else (xo_grid, xo_pdf)

    truth_1d_r = {i: grid_and_pdf_1d(i) for i in range(K)}

    truth_2d_r = {}
    for i in range(K):
        for j in range(i):
            xg, pxg = grid_and_pdf_1d(j)
            yg, pyg = grid_and_pdf_1d(i)
            same_pair = (i // 2 == j // 2) and (i % 2 != j % 2)
            if same_pair:
                Xg, Yg = np.meshgrid(xg, yg)
                pdf_2d = np.exp(-(b * (Yg - Xg ** 2) ** 2 + (Xg - a) ** 2))
            else:
                pdf_2d = np.outer(pyg, pxg)
            truth_2d_r[(i, j)] = (xg, yg, pdf_2d)

    for name, s in results.items():
        s_sub = np.asarray(s).reshape(-1, dim)[:, :K]
        fig = corner_plot(s_sub, labels=labels, truths=truths,
                          truth_1d=truth_1d_r, truth_2d=truth_2d_r,
                          title=f"{name} — first {K} dims")

    plt.show()
