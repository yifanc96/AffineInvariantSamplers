"""
run_funnel_ensemble_dr.py — Test ensemble DR (stretch + side) on Neal's funnel.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from affine_invariant_samplers import ensemble_dr as edr
from affine_invariant_samplers import stretch as stretch_mod
from affine_invariant_samplers import side as side_mod


# ══════════════════════════════════════════════════════════════════════════════
# Funnel
# ══════════════════════════════════════════════════════════════════════════════

def make_funnel(dim=5):
    d = dim - 1
    def log_prob(x):
        v = x[:, 0]
        xs = x[:, 1:]
        log_p_v = -0.5 * v**2 / 9.
        log_p_x = -0.5 * jnp.sum(xs**2 * jnp.exp(-v)[:, None], axis=1) \
                   - 0.5 * d * v
        return log_p_v + log_p_x
    return log_prob


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    dim       = 5
    n_chains  = 100
    n_samp    = 50000
    warmup    = 10000
    seed      = 42

    log_prob = make_funnel(dim=dim)
    init = jax.random.normal(jax.random.key(99), (n_chains, dim))

    print(f"Neal's Funnel  D={dim}  n_chains={n_chains}  n_samp={n_samp}  warmup={warmup}")
    print("=" * 80)

    results = {}

    # ────────── DR Stretch ──────────
    print("\n--- DR Stretch ---")
    s, info = edr.sampler_ensemble_dr_stretch(
        log_prob, init, n_samp, warmup=warmup, seed=seed)
    results["DR Stretch"] = (s, info)

    # ────────── DR Side ──────────
    print("\n--- DR Side ---")
    s, info = edr.sampler_ensemble_dr_side(
        log_prob, init, n_samp, warmup=warmup, seed=seed)
    results["DR Side"] = (s, info)

    # ────────── Plain Stretch (no DR) ──────────
    print("\n--- Stretch (no DR) ---")
    s, info = stretch_mod.sampler_stretch(
        log_prob, init, n_samp, warmup=warmup, seed=seed)
    results["Stretch"] = (s, info)

    # ────────── Plain Side (no DR) ──────────
    print("\n--- Side (no DR) ---")
    s, info = side_mod.sampler_side(
        log_prob, init, n_samp, warmup=warmup, seed=seed)
    results["Side"] = (s, info)

    # ══════════════════════════════════════════════════════════════════════════
    # Report
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print(f"{'Sampler':25s}  {'Accept':>8s}  {'v mean':>8s}  {'v var':>8s}  {'x var':>8s}")
    print("-" * 80)
    for name, (samps, info) in results.items():
        flat = np.asarray(samps.reshape(-1, dim))
        v = flat[:, 0]
        x_var = np.mean(np.var(flat[:, 1:], axis=0))
        acc = info.get("acceptance_rate", info.get("overall_rate", 0))
        print(f"{name:25s}  {acc:8.3f}  {np.mean(v):8.3f}  {np.var(v):8.3f}  {x_var:8.1f}")
    print("=" * 80)
    print("Target: v mean=0, v var=9, x_i var=exp(9/2)≈90.0")

    # ══════════════════════════════════════════════════════════════════════════
    # Plots
    # ══════════════════════════════════════════════════════════════════════════
    n_plot = len(results)
    fig, axes = plt.subplots(n_plot, 3, figsize=(15, 3.5 * n_plot))

    x_grid = np.linspace(-15, 15, 300)
    true_density = np.exp(-0.5 * x_grid**2 / 9.) / np.sqrt(2 * np.pi * 9.)

    for i, (name, (samps, info)) in enumerate(results.items()):
        flat = np.asarray(samps.reshape(-1, dim))
        v_samples = flat[:, 0]
        acc = info.get("acceptance_rate", info.get("overall_rate", 0))

        # Col 0: histogram of v
        bins = np.linspace(-15, 15, 60)
        axes[i, 0].hist(v_samples, bins=bins, density=True, alpha=0.7, label="samples")
        axes[i, 0].plot(x_grid, true_density, "r-", lw=2, label="N(0, 9)")
        axes[i, 0].set_xlabel("v")
        axes[i, 0].set_ylabel("density")
        axes[i, 0].set_title(f"{name}  (accept={acc:.3f})")
        axes[i, 0].legend(fontsize=8)
        axes[i, 0].set_xlim(-15, 15)

        # Col 1: scatter (v, x1)
        axes[i, 1].scatter(flat[::20, 0], flat[::20, 1], s=1, alpha=0.3)
        axes[i, 1].set_xlabel("v")
        axes[i, 1].set_ylabel("x₁")
        axes[i, 1].set_title(f"{name}: scatter (v, x₁)")

        # Col 2: trace of v (chain 0)
        v_trace = np.asarray(samps[:, 0, 0])
        axes[i, 2].plot(v_trace, lw=0.3)
        axes[i, 2].set_xlabel("iteration")
        axes[i, 2].set_ylabel("v")
        axes[i, 2].set_title(f"{name}: trace of v (chain 0)")

    plt.tight_layout()
    plt.savefig("ensemble_dr_funnel.png", dpi=150)
    plt.show()
    print("\nSaved ensemble_dr_funnel.png")