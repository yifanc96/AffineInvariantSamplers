"""
run_funnel_hmc.py — Test HMC samplers on Neal's funnel distribution.

Samplers tested:
  - NUTS          (identity mass matrix)
  - ChEES         (identity mass matrix, auto L)
  - PEACHES h-walk (ensemble preconditioned, walk move)
  - PEACHES h-side (ensemble preconditioned, side move)
  - PEANUTS h-walk (ensemble preconditioned NUTS, walk move)
  - PEANUTS h-side (ensemble preconditioned NUTS, side move)

Each sampler is tested with find_init_step_size=True (default) and
find_init_step_size=False (skip init search, let DA adapt from user step_size).

Neal's funnel:  v ~ N(0, 9),  x_i | v ~ N(0, exp(v)),  i = 1..d-1
  log p(v, x) = -v^2/18 - sum(x_i^2 / exp(v)) / 2 - (d-1)*v/2
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from affine_invariant_samplers import nuts as nuts
from affine_invariant_samplers import chess as chess
from affine_invariant_samplers import peaches as peaches
from affine_invariant_samplers import peanuts as peanuts


# ══════════════════════════════════════════════════════════════════════════════
# Funnel
# ══════════════════════════════════════════════════════════════════════════════

def make_funnel(dim=5):
    d = dim - 1

    def log_prob_batch(x):
        """(batch, dim) -> (batch,)"""
        v = x[:, 0]
        x_rest = x[:, 1:]
        return -0.5 * v**2 / 9. - 0.5 * jnp.sum(x_rest**2 / jnp.exp(v)[:, None], axis=1) - 0.5 * d * v

    return log_prob_batch


# ══════════════════════════════════════════════════════════════════════════════
# Run one sampler and collect stats
# ══════════════════════════════════════════════════════════════════════════════

def run_and_report(name, samples, info, dim):
    flat = np.asarray(samples.reshape(-1, dim))
    v = flat[:, 0]
    v_mean = np.mean(v)
    v_var  = np.var(v)
    acc    = float(info.get("acceptance_rate", info.get("mean_accept_prob", -1)))
    print(f"  {name:30s}  accept={acc:.3f}  v: mean={v_mean:.3f} var={v_var:.3f} (target: 0, 9)")
    return flat, acc


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    dim       = 5
    n_chains  = 100   # even, for peaches/peanuts
    n_samp    = 10000
    warmup    = 2000
    seed      = 123

    log_prob = make_funnel(dim=dim)
    init = 0.5 * jax.random.normal(jax.random.key(42), (n_chains, dim))

    print(f"Neal's Funnel  D={dim}  n_chains={n_chains}  n_samp={n_samp}  warmup={warmup}")
    print("=" * 80)

    results = {}  # name -> (flat_samples, accept_rate)

    # ────────── NUTS ──────────
    print("\n--- NUTS (find_init_step_size=True) ---")
    s, info = nuts.sampler_nuts(log_prob, init, n_samp, warmup=warmup,
                                step_size=0.1, find_init_step_size=True, seed=seed)
    results["NUTS (find_eps)"] = run_and_report("NUTS (find_eps)", s, info, dim)

    print("\n--- NUTS (find_init_step_size=False, eps=0.01) ---")
    s, info = nuts.sampler_nuts(log_prob, init, n_samp, warmup=warmup,
                                step_size=0.01, find_init_step_size=False, seed=seed)
    results["NUTS (no find)"] = run_and_report("NUTS (no find)", s, info, dim)

    # ────────── ChEES ──────────
    print("\n--- ChEES (find_init_step_size=True) ---")
    s, info = chess.sampler_chess(log_prob, init, n_samp, warmup=warmup,
                                  step_size=0.1, find_init_step_size=True, seed=seed)
    results["ChEES (find_eps)"] = run_and_report("ChEES (find_eps)", s, info, dim)

    print("\n--- ChEES (find_init_step_size=False, eps=0.01) ---")
    s, info = chess.sampler_chess(log_prob, init, n_samp, warmup=warmup,
                                  step_size=0.01, find_init_step_size=False, seed=seed)
    results["ChEES (no find)"] = run_and_report("ChEES (no find)", s, info, dim)

    # ────────── PEACHES h-walk ──────────
    print("\n--- PEACHES h-walk (find_init_step_size=True) ---")
    s, info = peaches.sampler_peaches(log_prob, init, n_samp, warmup=warmup,
                                       move="h-walk", step_size=0.05,
                                       find_init_step_size=True, seed=seed)
    results["PEACHES walk (find)"] = run_and_report("PEACHES walk (find)", s, info, dim)

    print("\n--- PEACHES h-walk (find_init_step_size=False, eps=0.001) ---")
    s, info = peaches.sampler_peaches(log_prob, init, n_samp, warmup=warmup,
                                       move="h-walk", step_size=0.001,
                                       find_init_step_size=False, seed=seed)
    results["PEACHES walk (no find)"] = run_and_report("PEACHES walk (no find)", s, info, dim)

    # ────────── PEACHES h-side ──────────
    print("\n--- PEACHES h-side (find_init_step_size=True) ---")
    s, info = peaches.sampler_peaches(log_prob, init, n_samp, warmup=warmup,
                                       move="h-side", step_size=0.05,
                                       find_init_step_size=True, seed=seed)
    results["PEACHES side (find)"] = run_and_report("PEACHES side (find)", s, info, dim)

    print("\n--- PEACHES h-side (find_init_step_size=False, eps=0.01) ---")
    s, info = peaches.sampler_peaches(log_prob, init, n_samp, warmup=warmup,
                                       move="h-side", step_size=0.01,
                                       find_init_step_size=False, seed=seed)
    results["PEACHES side (no find)"] = run_and_report("PEACHES side (no find)", s, info, dim)

    # ────────── PEANUTS h-walk ──────────
    print("\n--- PEANUTS h-walk (find_init_step_size=True) ---")
    s, info = peanuts.sampler_peanuts(log_prob, init, n_samp, warmup=warmup,
                                       move="h-walk", step_size=0.05,
                                       find_init_step_size=True, seed=seed)
    results["PEANUTS walk (find)"] = run_and_report("PEANUTS walk (find)", s, info, dim)

    print("\n--- PEANUTS h-walk (find_init_step_size=False, eps=0.001) ---")
    s, info = peanuts.sampler_peanuts(log_prob, init, n_samp, warmup=warmup,
                                       move="h-walk", step_size=0.001,
                                       find_init_step_size=False, seed=seed)
    results["PEANUTS walk (no find)"] = run_and_report("PEANUTS walk (no find)", s, info, dim)

    # ────────── PEANUTS h-side ──────────
    print("\n--- PEANUTS h-side (find_init_step_size=True) ---")
    s, info = peanuts.sampler_peanuts(log_prob, init, n_samp, warmup=warmup,
                                       move="h-side", step_size=0.05,
                                       find_init_step_size=True, seed=seed)
    results["PEANUTS side (find)"] = run_and_report("PEANUTS side (find)", s, info, dim)

    print("\n--- PEANUTS h-side (find_init_step_size=False, eps=0.01) ---")
    s, info = peanuts.sampler_peanuts(log_prob, init, n_samp, warmup=warmup,
                                       move="h-side", step_size=0.01,
                                       find_init_step_size=False, seed=seed)
    results["PEANUTS side (no find)"] = run_and_report("PEANUTS side (no find)", s, info, dim)

    # ══════════════════════════════════════════════════════════════════════════
    # Plots
    # ══════════════════════════════════════════════════════════════════════════

    # Select a subset of results for plotting (prefer "no find" variants)
    plot_keys = [
        "NUTS (no find)",
        "ChEES (no find)",
        "PEACHES walk (no find)",
        "PEACHES side (no find)",
        "PEANUTS walk (no find)",
        "PEANUTS side (no find)",
    ]
    # Fallback: if "no find" failed badly, also show "find" version
    plot_keys_find = [
        "NUTS (find_eps)",
        "ChEES (find_eps)",
        "PEACHES walk (find)",
        "PEACHES side (find)",
        "PEANUTS walk (find)",
        "PEANUTS side (find)",
    ]

    n_plot = len(plot_keys)
    fig, axes = plt.subplots(n_plot, 3, figsize=(15, 3.5 * n_plot))

    x_grid = np.linspace(-15, 15, 300)
    true_density = np.exp(-0.5 * x_grid**2 / 9.) / np.sqrt(2 * np.pi * 9.)

    for i, (key, key_find) in enumerate(zip(plot_keys, plot_keys_find)):
        flat, acc = results.get(key, (None, 0))
        if flat is None:
            continue

        v_samples = flat[:, 0]

        # Column 0: histogram of v
        bins = np.linspace(-15, 15, 60)
        axes[i, 0].hist(v_samples, bins=bins, density=True, alpha=0.7, label="samples")
        axes[i, 0].plot(x_grid, true_density, "r-", lw=2, label="N(0, 9)")
        axes[i, 0].set_xlabel("v")
        axes[i, 0].set_ylabel("density")
        axes[i, 0].set_title(f"{key}  (accept={acc:.3f})")
        axes[i, 0].legend(fontsize=8)
        axes[i, 0].set_xlim(-15, 15)

        # Column 1: scatter (v, x1)
        axes[i, 1].scatter(flat[::10, 0], flat[::10, 1], s=1, alpha=0.3)
        axes[i, 1].set_xlabel("v")
        axes[i, 1].set_ylabel("x₁")
        axes[i, 1].set_title(f"{key}: scatter (v, x₁)")

        # Column 2: trace of v (first few thousand samples from chain 0)
        # Reconstruct trace from flat: n_samp samples per chain
        n_trace = min(5000, len(flat) // n_chains)
        v_trace = flat[:n_trace, 0]  # approximate trace from first chain's worth
        axes[i, 2].plot(v_trace, lw=0.3)
        axes[i, 2].set_xlabel("sample index")
        axes[i, 2].set_ylabel("v")
        axes[i, 2].set_title(f"{key}: trace of v")

    plt.tight_layout()
    plt.savefig("hmc_funnel_tests.png", dpi=150)
    plt.show()
    print("\nSaved hmc_funnel_tests.png")

    # ══════════════════════════════════════════════════════════════════════════
    # Summary table
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print(f"{'Sampler':35s}  {'Accept':>8s}  {'v mean':>8s}  {'v var':>8s}")
    print("-" * 80)
    for key in list(results.keys()):
        flat, acc = results[key]
        v = flat[:, 0]
        print(f"{key:35s}  {acc:8.3f}  {np.mean(v):8.3f}  {np.var(v):8.3f}")
    print("=" * 80)
    print("Target: v mean=0.000, v var=9.000")
    print("Done.")