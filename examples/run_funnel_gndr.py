"""
run_funnel.py — Test GN-DR on Neal's funnel and Rosenbrock distributions.

Neal's funnel:  v ~ N(0, 9),  x_i | v ~ N(0, exp(v)),  i = 1..d-1
  log p(v, x) = -v^2/18 - sum(x_i^2 / exp(v)) / 2 - (d-1)*v/2
  GN residual:  r(x) = [v/3, x_i/exp(v/2)]

Rosenbrock:  log p(x) = -sum[ b*(x_{i+1} - x_i^2)^2 + (x_i - a)^2 ]
  GN residual:  r(x) = [sqrt(b)*(x_{i+1} - x_i^2), (x_i - a)]
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from affine_invariant_samplers import gndr
# ══════════════════════════════════════════════════════════════════════════════
# Funnel
# ══════════════════════════════════════════════════════════════════════════════

def make_funnel(dim=5):
    d = dim - 1

    def log_prob(x):
        v = x[0]
        x_rest = x[1:]
        return -0.5 * v**2 / 9. - 0.5 * jnp.sum(x_rest**2 / jnp.exp(v)) - 0.5 * d * v

    def residual(x):
        v = x[0]
        x_rest = x[1:]
        return jnp.concatenate([jnp.array([v / 3.]), x_rest / jnp.exp(v / 2.)])

    return log_prob, residual


# ══════════════════════════════════════════════════════════════════════════════
# Rosenbrock
# ══════════════════════════════════════════════════════════════════════════════

def make_rosenbrock(dim=10, a=1.0, b=100.0):
    """
    Rosenbrock:  log p(x) = -sum[ b*(x_{i+1} - x_i^2)^2 + (x_i - a)^2 ]
    Residual:    r(x) = [sqrt(b)*(x_{i+1} - x_i^2), (x_i - a)]  for i = 0,2,4,...
    """
    sb = jnp.sqrt(b)

    def log_prob(x):
        x_even = x[::2]
        x_odd  = x[1::2]
        return -(b * jnp.sum((x_odd - x_even**2)**2)
                 + jnp.sum((x_even - a)**2))

    def residual(x):
        x_even = x[::2]
        x_odd  = x[1::2]
        return jnp.concatenate([sb * (x_odd - x_even**2), x_even - a])

    return log_prob, residual


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ────────── Funnel ──────────
    dim_f = 5
    n_chains_f = 100

    log_prob_f, residual_f = make_funnel(dim=dim_f)
    init_f = 0.5 * jax.random.normal(jax.random.key(42), (n_chains_f, dim_f))

    print(f"Funnel  D={dim_f}  n_chains={n_chains_f}")
    print("=" * 60)

    s_f, info_f = gndr.sampler_gndr(
        log_prob_f, init_f,
        num_samples=10000,
        warmup=2000,
        step_size=0.1,
        n_try=3,
        shrink=0.2,
        residual_fn=residual_f,
        seed=123,
    )

    flat_f = np.asarray(s_f.reshape(-1, dim_f))
    print(f"\nv:  mean={np.mean(flat_f[:,0]):.3f}  var={np.var(flat_f[:,0]):.3f}  (target: mean=0, var=9)")
    print(f"v range: [{np.min(flat_f[:,0]):.2f}, {np.max(flat_f[:,0]):.2f}]")
    for i in range(1, min(4, dim_f)):
        print(f"x{i}: mean={np.mean(flat_f[:,i]):.3f}  var={np.var(flat_f[:,i]):.2f}")

    # ────────── Rosenbrock ──────────
    dim_r = 10
    n_chains_r = 100

    log_prob_r, residual_r = make_rosenbrock(dim=dim_r)
    init_r = jax.random.normal(jax.random.key(42), (n_chains_r, dim_r))

    print(f"\nRosenbrock  D={dim_r}  a=1  b=100  n_chains={n_chains_r}")
    print("=" * 60)

    s_r, info_r = gndr.sampler_gndr(
        log_prob_r, init_r,
        num_samples=10000,
        warmup=2000,
        step_size=0.01,
        n_try=3,
        shrink=0.2,
        residual_fn=residual_r,
        seed=123,
    )

    flat_r = np.asarray(s_r.reshape(-1, dim_r))
    me = np.mean(flat_r[:, ::2]);  ve = np.mean(np.var(flat_r[:, ::2], axis=0))
    mo = np.mean(flat_r[:, 1::2]); vo = np.mean(np.var(flat_r[:, 1::2], axis=0))
    print(f"\nx_even: mean={me:.3f}  var={ve:.4f}  (target: mean=1, var=0.5)")
    print(f"x_odd:  mean={mo:.3f}  var={vo:.4f}  (target: mean=1.5, var~2.505)")

    # ────────── Plots ──────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Row 1: Funnel
    v_samples = flat_f[:, 0]
    bins = np.linspace(-15, 15, 60)
    axes[0, 0].hist(v_samples, bins=bins, density=True, alpha=0.7, label="samples")
    x_grid = np.linspace(-15, 15, 300)
    true_density = np.exp(-0.5 * x_grid**2 / 9.) / np.sqrt(2 * np.pi * 9.)
    axes[0, 0].plot(x_grid, true_density, "r-", lw=2, label="N(0, 9)")
    axes[0, 0].set_xlabel("v")
    axes[0, 0].set_ylabel("density")
    axes[0, 0].set_title("Funnel: marginal of v")
    axes[0, 0].legend()

    axes[0, 1].scatter(flat_f[::10, 0], flat_f[::10, 1], s=1, alpha=0.3)
    axes[0, 1].set_xlabel("v")
    axes[0, 1].set_ylabel("x₁")
    axes[0, 1].set_title("Funnel: scatter (v, x₁)")

    v_trace = np.asarray(s_f[:, 0, 0])
    axes[0, 2].plot(v_trace, lw=0.3)
    axes[0, 2].set_xlabel("iteration")
    axes[0, 2].set_ylabel("v")
    axes[0, 2].set_title("Funnel: trace of v (chain 0)")

    # Row 2: Rosenbrock
    axes[1, 0].scatter(flat_r[::10, 0], flat_r[::10, 1], s=1, alpha=0.3)
    axes[1, 0].set_xlabel("x₀")
    axes[1, 0].set_ylabel("x₁")
    axes[1, 0].set_title("Rosenbrock: scatter (x₀, x₁)")

    axes[1, 1].hist(flat_r[:, 0], bins=50, density=True, alpha=0.7, label="x₀ samples")
    axes[1, 1].axvline(1.0, color="r", ls="--", lw=2, label="true mean=1")
    axes[1, 1].set_xlabel("x₀")
    axes[1, 1].set_ylabel("density")
    axes[1, 1].set_title("Rosenbrock: marginal of x₀")
    axes[1, 1].legend()

    x0_trace = np.asarray(s_r[:, 0, 0])
    axes[1, 2].plot(x0_trace, lw=0.3)
    axes[1, 2].axhline(1.0, color="r", ls="--", lw=1)
    axes[1, 2].set_xlabel("iteration")
    axes[1, 2].set_ylabel("x₀")
    axes[1, 2].set_title("Rosenbrock: trace of x₀ (chain 0)")

    plt.tight_layout()
    plt.savefig("gndr_tests.png", dpi=150)
    plt.show()
    print("\nSaved gndr_tests.png")