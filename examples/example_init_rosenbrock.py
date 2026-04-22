"""
example_init_rosenbrock.py — Initialization helpers on 10-D Rosenbrock.

Demonstrates the three initialization utilities from
``affine_invariant_samplers.init`` on the banana target:

    * find_map                 — single-start BFGS
    * find_map_restarts        — multi-start BFGS via vmap
    * init_ensemble_from_map   — MAP + Laplace → ensemble

and then runs `sampler_peaches` from three different starting ensembles
(isotropic random / jitter around MAP / Laplace-seeded) so you can see the
effect of a good initialization on warm-up and sample quality.

Same target as `example_rosenbrock.py`:  (a, b) = (1, 100), D = 10.
True MAP is (1, 1, ..., 1) with -log_prob = 0.  The Laplace Hessian at the
MAP has a very soft direction along the banana valley — the Laplace-seeded
ensemble inherits that spread automatically.
"""

from __future__ import annotations

import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from affine_invariant_samplers import (
    sampler_peaches,
    find_map,
    find_map_restarts,
    init_ensemble_from_map,
    effective_sample_size,
)


# ──────────────────────────────────────────────────────────────────────────────
# Target:  10-D Rosenbrock
# ──────────────────────────────────────────────────────────────────────────────

def make_rosenbrock(dim=10, a=1.0, b=100.0):
    assert dim % 2 == 0

    def log_prob_batched(x):                    # (batch, D) -> (batch,)
        xe, xo = x[:, ::2], x[:, 1::2]
        return -(b * jnp.sum((xo - xe ** 2) ** 2, axis=1)
                 + jnp.sum((xe - a) ** 2, axis=1))

    def log_prob_single(x):                     # (D,) -> scalar
        xe, xo = x[::2], x[1::2]
        return -(b * jnp.sum((xo - xe ** 2) ** 2)
                 + jnp.sum((xe - a) ** 2))

    return log_prob_batched, log_prob_single


# ──────────────────────────────────────────────────────────────────────────────
# Report helper
# ──────────────────────────────────────────────────────────────────────────────

def _report(name, samples, info, elapsed):
    flat = jnp.asarray(samples).reshape(-1, samples.shape[-1])
    xe, xo = flat[:, ::2], flat[:, 1::2]
    me, ve = float(jnp.mean(xe)), float(jnp.mean(jnp.var(xe, axis=0)))
    mo, vo = float(jnp.mean(xo)), float(jnp.mean(jnp.var(xo, axis=0)))
    ess    = effective_sample_size(samples)
    print(f"  {name:<26s}  x_e mean={me:5.2f} var={ve:5.2f}   "
          f"x_o mean={mo:5.2f} var={vo:5.2f}   "
          f"min_ESS={float(ess.min()):7.1f}   time={elapsed:5.1f}s")


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dim      = 10
    a, b     = 1.0, 100.0
    n_chains = 100
    n_samp   = 3000
    warmup   = 1000
    seed     = 123

    log_prob, log_prob_single = make_rosenbrock(dim=dim, a=a, b=b)

    print(f"Rosenbrock initialization demo  D={dim}  (a, b) = ({a}, {b})")
    print("=" * 92)

    # ──────────────────────────────────────────────────────────────────────
    # Tier 1 — single-start BFGS
    # ──────────────────────────────────────────────────────────────────────
    print("\n--- Tier 1: find_map (single-start BFGS from zero) ---")
    r1 = find_map(log_prob_single, jnp.zeros(dim), maxiter=500, verbose=True)
    print(f"   x_MAP ≈ {np.asarray(r1.x).round(3)}")
    print(f"   (true MAP is (1, 1, ..., 1);  BFGS from zero on Rosenbrock often")
    print(f"    stalls — Tier 2 with restarts fixes that.)")

    # ──────────────────────────────────────────────────────────────────────
    # Tier 2 — multi-start BFGS
    # ──────────────────────────────────────────────────────────────────────
    print("\n--- Tier 2: find_map_restarts (8 random restarts) ---")
    key_restart = jax.random.key(0)
    x0_batch    = jax.random.normal(key_restart, (8, dim))
    r2 = find_map_restarts(log_prob_single, x0_batch, maxiter=500, verbose=True)
    print(f"   x_MAP ≈ {np.asarray(r2.x).round(3)}")

    # ──────────────────────────────────────────────────────────────────────
    # Tier 3 — MAP + Laplace ensemble
    # ──────────────────────────────────────────────────────────────────────
    print("\n--- Tier 3: init_ensemble_from_map (Laplace-seeded ensemble) ---")
    init_laplace, r3 = init_ensemble_from_map(
        log_prob_single, jnp.zeros(dim),
        n_chains=n_chains, seed=seed,
        n_restarts=8, scale=1.0, verbose=True,
    )
    print(f"   ensemble shape: {init_laplace.shape}")
    print(f"   per-dim std: {np.asarray(init_laplace.std(axis=0)).round(3)}")
    print(f"   (large spread along the banana valley — Laplace inherits the")
    print(f"    soft Hessian eigenvector at the MAP.)")

    # ──────────────────────────────────────────────────────────────────────
    # Sampling: peaches from three different init styles
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("Running sampler_peaches from three initializations:")
    print("=" * 92)

    # (a) Isotropic random init — the usual "no-info" start.
    init_random = jax.random.normal(jax.random.key(42), (n_chains, dim))

    # (b) Tight jitter around the MAP — under-dispersed on purpose to show
    #     what `find_init_step_size` gets wrong on a concentrated init.
    init_tight  = r2.x[None, :] + 0.01 * jax.random.normal(
        jax.random.key(43), (n_chains, dim))

    # (c) Laplace-seeded ensemble from Tier 3.
    inits = {
        "isotropic N(0, I)":      init_random,
        "tight jitter @ MAP":     init_tight,
        "Laplace @ MAP (Tier 3)": init_laplace,
    }

    results = {}
    for name, init in inits.items():
        t0 = time.time()
        s, info = sampler_peaches(log_prob, init, n_samp, warmup=warmup,
                                   step_size=0.01, seed=seed, verbose=True)
        _report(name, s, info, time.time() - t0)
        results[name] = s

    print("=" * 92)
    print("Target: x_e mean=1.00, var=0.50   x_o mean=1.50, var≈2.505")
    print("\nTakeaways:")
    print("  * Tier 1 from zero often fails on Rosenbrock — Tier 2 fixes it.")
    print("  * Tight jitter around MAP makes `find_init_step_size` overshoot.")
    print("  * Laplace init gives a well-conditioned start on the right scale.")

    # ──────────────────────────────────────────────────────────────────────
    # Plot: starting ensembles in (x₀, x₁) overlaid on true contours
    # ──────────────────────────────────────────────────────────────────────
    xr = np.linspace(-2.5, 2.5, 400)
    yr = np.linspace(-1.5, 6.0, 400)
    gx, gy = np.meshgrid(xr, yr)
    true_density = np.exp(-(b * (gy - gx ** 2) ** 2 + (gx - a) ** 2))

    fig, axes = plt.subplots(1, len(inits), figsize=(4.4 * len(inits), 4.2),
                              sharex=True, sharey=True)
    for ax, (name, init) in zip(axes, inits.items()):
        init_np = np.asarray(init)
        ax.contour(gx, gy, true_density, levels=8, colors="k",
                   linewidths=0.8, alpha=0.6)
        ax.scatter(init_np[:, 0], init_np[:, 1], s=12, color="C3",
                   alpha=0.7, edgecolor="none", label="init ensemble")
        ax.axvline(a, color="k", lw=0.5, ls=":")
        ax.axhline(a ** 2, color="k", lw=0.5, ls=":")
        ax.set_xlim(xr[0], xr[-1]); ax.set_ylim(yr[0], yr[-1])
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("x₀ (even)")
    axes[0].set_ylabel("x₁ (odd)")
    fig.suptitle("Starting ensembles on Rosenbrock  (black = true contours, "
                 "dotted = MAP)", y=0.99)
    fig.tight_layout()

    # ──────────────────────────────────────────────────────────────────────
    # Plot: resulting samples (same projection)
    # ──────────────────────────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, len(results), figsize=(4.4 * len(results), 4.2),
                                sharex=True, sharey=True)
    for ax, (name, s) in zip(axes2, results.items()):
        flat = np.asarray(s).reshape(-1, dim)
        ax.hist2d(flat[:, 0], flat[:, 1], bins=80,
                  range=[[xr[0], xr[-1]], [yr[0], yr[-1]]],
                  cmap="Blues", density=True)
        ax.contour(gx, gy, true_density, levels=8, colors="k",
                   linewidths=0.8, alpha=0.7)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("x₀ (even)")
    axes2[0].set_ylabel("x₁ (odd)")
    fig2.suptitle("sampler_peaches samples (black = true contours)", y=0.99)
    fig2.tight_layout()

    plt.show()
