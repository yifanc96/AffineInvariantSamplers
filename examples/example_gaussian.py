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
import numpy as np
import matplotlib.pyplot as plt

from affine_invariant_samplers import (
    sampler_stretch,
    sampler_langevin_walk,
    sampler_kalman_move,
    sampler_peaches,
    effective_sample_size,
)
from affine_invariant_samplers.plotting import corner_plot


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

def _report(name, samples, cov, info, elapsed):
    flat = jnp.asarray(samples).reshape(-1, samples.shape[-1])
    var_est  = jnp.var(flat, axis=0)
    var_true = jnp.diag(cov)
    rel_err  = float(jnp.mean(jnp.abs(var_est - var_true) / var_true))
    ess      = effective_sample_size(samples)
    accept   = info["acceptance_rate"]
    grads    = info.get("n_grad_evals")
    grads_s  = f"{grads:>10d}" if grads is not None else f"{'–':>10s}"
    print(f"  {name:<20s}  rel_err(var)={rel_err:6.3f}   "
          f"accept={accept:5.3f}   min_ESS={float(ess.min()):7.1f}   "
          f"grad_evals={grads_s}   time={elapsed:5.1f}s")


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dim      = 50
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

    results = {}

    t0 = time.time()
    s, info = sampler_stretch(log_prob, init, n_samp, warmup=warmup, seed=seed,
                              verbose=True)
    _report("stretch",       s, cov, info, time.time() - t0)
    results["stretch"] = s

    # For ensemble-preconditioned Langevin (langevin_walk, kalman_move) under
    # an under-dispersed init (N(0, I) vs diag(cov)∈[45, 402]), the default
    # `find_init_step_size` heuristic over-estimates h by latching onto the
    # tiny initial ensemble covariance.  We disable it and pass a sensible
    # starting step size; DA refines from there.
    t0 = time.time()
    s, info = sampler_langevin_walk(log_prob, init, n_samp, warmup=warmup,
                                     step_size=0.3, find_init_step_size=False,
                                     seed=seed, verbose=True)
    _report("langevin_walk", s, cov, info, time.time() - t0)
    results["langevin_walk"] = s

    t0 = time.time()
    s, info = sampler_kalman_move(log_prob, lambda x: x, prec, init, n_samp,
                                   warmup=warmup, step_size=0.3,
                                   find_init_step_size=False, seed=seed,
                                   verbose=True)
    _report("kalman_move",   s, cov, info, time.time() - t0)
    results["kalman_move"] = s

    t0 = time.time()
    s, info = sampler_peaches(log_prob, init, n_samp, warmup=warmup,
                               step_size=0.01, seed=seed, verbose=True)
    _report("peaches",       s, cov, info, time.time() - t0)
    results["peaches"] = s

    print("=" * 80)
    print(f"Target: diag(cov) spans [{float(jnp.diag(cov).min()):.2f}, "
          f"{float(jnp.diag(cov).max()):.2f}]  (ratio = kappa = {kappa})")

    # ──────────────────────────────────────────────────────────────────────
    # Plots
    # ──────────────────────────────────────────────────────────────────────

    # 2D projection onto the top-2 principal axes (largest eigenvalues).
    # These are the hardest directions to sample.
    cov_np = np.asarray(cov)
    eigvals_np, eigvecs_np = np.linalg.eigh(cov_np)
    pc1, pc2 = eigvecs_np[:, -1], eigvecs_np[:, -2]
    l1, l2   = eigvals_np[-1], eigvals_np[-2]

    # True 2D contour in PC-space is N(0, diag(l1, l2))
    xmax = 3.5 * np.sqrt(l1);  ymax = 3.5 * np.sqrt(l2)
    gx, gy = np.meshgrid(np.linspace(-xmax, xmax, 200),
                         np.linspace(-ymax, ymax, 200))
    true_density = np.exp(-0.5 * (gx ** 2 / l1 + gy ** 2 / l2))

    fig_c, axes_c = plt.subplots(1, len(results), figsize=(4 * len(results), 4.2),
                                  sharex=True, sharey=True)
    for ax, (name, s) in zip(axes_c, results.items()):
        flat = np.asarray(s).reshape(-1, dim)
        proj1, proj2 = flat @ pc1, flat @ pc2
        ax.hist2d(proj1, proj2, bins=80, range=[[-xmax, xmax], [-ymax, ymax]],
                  cmap="Blues", density=True)
        ax.contour(gx, gy, true_density, levels=6, colors="k",
                   linewidths=0.8, alpha=0.7)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("PC1 (λ = {:.0f})".format(l1))
    axes_c[0].set_ylabel("PC2 (λ = {:.0f})".format(l2))
    fig_c.suptitle("Top-2 principal-axis projection  (black = true 2-σ contours)",
                   y=0.99)
    fig_c.tight_layout()

    # Per-method corner plots on the first K dims, with analytical marginals
    # (the joint is N(0, cov[:K, :K])) overlaid.
    K = 5
    labels = [f"x{i}" for i in range(K)]
    cov_K = cov_np[:K, :K]
    sigs  = np.sqrt(np.diag(cov_K))

    truth_1d = {}
    for i in range(K):
        xg = np.linspace(-3.5 * sigs[i], 3.5 * sigs[i], 200)
        truth_1d[i] = (xg, np.exp(-0.5 * xg ** 2 / cov_K[i, i]) /
                           np.sqrt(2 * np.pi * cov_K[i, i]))

    truth_2d = {}
    for i in range(K):
        for j in range(i):
            C  = cov_K[np.ix_([i, j], [i, j])]
            Ci = np.linalg.inv(C)
            xg = np.linspace(-3.5 * sigs[j], 3.5 * sigs[j], 150)
            yg = np.linspace(-3.5 * sigs[i], 3.5 * sigs[i], 150)
            Xg, Yg = np.meshgrid(xg, yg)
            q = Ci[0, 0] * Xg ** 2 + 2 * Ci[0, 1] * Xg * Yg + Ci[1, 1] * Yg ** 2
            truth_2d[(i, j)] = (xg, yg, np.exp(-0.5 * q))

    for name, s in results.items():
        s_sub = np.asarray(s).reshape(-1, dim)[:, :K]
        fig = corner_plot(s_sub, labels=labels, truths=[0.0] * K,
                          truth_1d=truth_1d, truth_2d=truth_2d,
                          title=f"{name} — first {K} dims")

    plt.show()
