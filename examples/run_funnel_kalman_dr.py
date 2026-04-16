"""
run_funnel_kalman_dr.py — Test Kalman-DR on Neal's funnel distribution.

The Kalman-DR is derivative-free: it only needs a forward model G(x) and
data-space precision M, no gradients.

For the funnel, we use the "residual" decomposition:
  log p(v, x) = -0.5 ||r(v,x)||^2 - (d-1)*v/2
  r(v, x) = [v/3, x_i / exp(v/2)]

Forward model G(x) = r(x),  data-space precision M = I.
The linear term -(d-1)*v/2 has zero Hessian so the GN/Kalman approximation
captures all curvature.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from affine_invariant_samplers import kalman_dr as kalman_dr
from affine_invariant_samplers import kalman_move as kalman


# ══════════════════════════════════════════════════════════════════════════════
# Funnel
# ══════════════════════════════════════════════════════════════════════════════

def make_funnel(dim=5):
    d = dim - 1

    def log_prob_single(x):
        v = x[0]
        x_rest = x[1:]
        return -0.5 * v**2 / 9. - 0.5 * jnp.sum(x_rest**2 / jnp.exp(v)) - 0.5 * d * v

    def log_prob_batch(x):
        return jax.vmap(log_prob_single)(x)

    def forward_single(x):
        """G(x) = r(x), the residual vector."""
        v = x[0]
        x_rest = x[1:]
        return jnp.concatenate([jnp.array([v / 3.]), x_rest / jnp.exp(v / 2.)])

    def forward_batch(x):
        return jax.vmap(forward_single)(x)

    # data-space precision M = I (since log_prob ≈ -0.5 ||r||^2 up to linear term)
    M = jnp.eye(dim)

    return log_prob_batch, forward_batch, M


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    dim       = 5
    n_chains  = 100
    n_samp    = 50000
    warmup    = 10000
    seed      = 123

    log_prob, forward_fn, M = make_funnel(dim=dim)
    init = 0.5 * jax.random.normal(jax.random.key(42), (n_chains, dim))

    print(f"Neal's Funnel  D={dim}  n_chains={n_chains}")
    print("=" * 70)

    # ────────── Kalman-DR (n_try=3) ──────────
    print("\n--- Kalman-DR (n_try=3) ---")
    s_dr, info_dr = kalman_dr.sampler_kalman_dr(
        log_prob, forward_fn, M, init,
        num_samples=n_samp,
        warmup=warmup,
        step_size=0.1,
        n_try=3,
        shrink=0.2,
        seed=seed,
        subset_size=2
    )
    ### side-move Kalman-DR with n_try=3 shows good exploration of v, with mean and variance close to target (0, 9).
    flat_dr = np.asarray(s_dr.reshape(-1, dim))
    v_dr = flat_dr[:, 0]
    print(f"  v:  mean={np.mean(v_dr):.3f}  var={np.var(v_dr):.3f}  (target: 0, 9)")
    print(f"  v range: [{np.min(v_dr):.2f}, {np.max(v_dr):.2f}]")

    # ────────── Kalman-DR (n_try=1, plain MH for comparison) ──────────
    print("\n--- Kalman-DR (n_try=1, plain MH) ---")
    s_mh, info_mh = kalman_dr.sampler_kalman_dr(
        log_prob, forward_fn, M, init,
        num_samples=n_samp,
        warmup=warmup,
        step_size=0.1,
        n_try=1,
        seed=seed,
    )

    flat_mh = np.asarray(s_mh.reshape(-1, dim))
    v_mh = flat_mh[:, 0]
    print(f"  v:  mean={np.mean(v_mh):.3f}  var={np.var(v_mh):.3f}  (target: 0, 9)")
    print(f"  v range: [{np.min(v_mh):.2f}, {np.max(v_mh):.2f}]")

    # ────────── Kalman move (no DR, original) ──────────
    print("\n--- Kalman move (no DR) ---")
    s_kw, info_kw = kalman.sampler_kalman_move(
        log_prob, forward_fn, M, init,
        num_samples=n_samp,
        warmup=warmup,
        step_size=0.05,
        seed=seed,
    )

    flat_kw = np.asarray(s_kw.reshape(-1, dim))
    v_kw = flat_kw[:, 0]
    print(f"  v:  mean={np.mean(v_kw):.3f}  var={np.var(v_kw):.3f}  (target: 0, 9)")
    print(f"  v range: [{np.min(v_kw):.2f}, {np.max(v_kw):.2f}]")

    # ══════════════════════════════════════════════════════════════════════════
    # Plots
    # ══════════════════════════════════════════════════════════════════════════

    results = {
        "Kalman-DR (n_try=3)": (flat_dr, info_dr, s_dr),
        "Kalman-DR (n_try=1)": (flat_mh, info_mh, s_mh),
        "Kalman walk (no DR)": (flat_kw, info_kw, s_kw),
    }

    n_plot = len(results)
    fig, axes = plt.subplots(n_plot, 3, figsize=(15, 3.5 * n_plot))

    x_grid = np.linspace(-15, 15, 300)
    true_density = np.exp(-0.5 * x_grid**2 / 9.) / np.sqrt(2 * np.pi * 9.)

    for i, (name, (flat, info, samps)) in enumerate(results.items()):
        v_samples = flat[:, 0]
        acc = info.get("acceptance_rate", 0)
        s1  = info.get("stage1_rate", acc)

        # Col 0: histogram of v
        bins = np.linspace(-15, 15, 60)
        axes[i, 0].hist(v_samples, bins=bins, density=True, alpha=0.7, label="samples")
        axes[i, 0].plot(x_grid, true_density, "r-", lw=2, label="N(0, 9)")
        axes[i, 0].set_xlabel("v")
        axes[i, 0].set_ylabel("density")
        axes[i, 0].set_title(f"{name}  (accept={acc:.3f}, s1={s1:.3f})")
        axes[i, 0].legend(fontsize=8)
        axes[i, 0].set_xlim(-15, 15)

        # Col 1: scatter (v, x1)
        axes[i, 1].scatter(flat[::10, 0], flat[::10, 1], s=1, alpha=0.3)
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
    plt.savefig("kalman_dr_funnel.png", dpi=150)
    plt.show()
    print("\nSaved kalman_dr_funnel.png")

    # ══════════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"{'Sampler':30s}  {'Accept':>8s}  {'S1':>8s}  {'v mean':>8s}  {'v var':>8s}")
    print("-" * 70)
    for name, (flat, info, _) in results.items():
        acc = info.get("acceptance_rate", 0)
        s1  = info.get("stage1_rate", acc)
        v = flat[:, 0]
        print(f"{name:30s}  {acc:8.3f}  {s1:8.3f}  {np.mean(v):8.3f}  {np.var(v):8.3f}")
    print("=" * 70)
    print("Target: v mean=0.000, v var=9.000")