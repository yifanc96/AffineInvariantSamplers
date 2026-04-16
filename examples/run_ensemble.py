"""
run_ensemble.py — Test derivative-free and gradient-based ensemble samplers
on Gaussian and Rosenbrock distributions.

Samplers tested:
  Derivative-free:
    - stretch       Goodman & Weare (2010) stretch move + DA
    - side          Affine-invariant side move + DA
    - walk          Walk move with k-subset + DA
    - ensemble_dr_stretch  Delayed rejection stretch move (2-stage) + DA
    - ensemble_dr_side     Delayed rejection side move (2-stage) + DA

  Gradient-based:
    - langevin_walk Langevin walk move (MALA-like, ensemble-preconditioned)
    - kalman_move   Ensemble Kalman move (requires forward model)
    - gndr          Gauss-Newton proposal Langevin with multi-stage DR
"""

import jax
import jax.numpy as jnp

# Import samplers
from affine_invariant_samplers import stretch
from affine_invariant_samplers import side
from affine_invariant_samplers import walk
from affine_invariant_samplers import langevin_walk
from affine_invariant_samplers import kalman_move
from affine_invariant_samplers import ensemble_dr
from affine_invariant_samplers import gndr
# ══════════════════════════════════════════════════════════════════════════════
# Test distributions
# ══════════════════════════════════════════════════════════════════════════════

def make_gaussian(dim=10, kappa=100., seed=0):
    """Ill-conditioned Gaussian.  Returns (log_prob, true_cov, precision)."""
    eigvals = jnp.logspace(0, jnp.log10(kappa), dim)
    Q, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(seed), (dim, dim)))
    cov  = Q @ jnp.diag(eigvals) @ Q.T
    prec = Q @ jnp.diag(1. / eigvals) @ Q.T
    L_prec = jnp.linalg.cholesky(prec)
    def log_prob(x):
        return -0.5 * jnp.sum((x @ prec) * x, axis=-1)
    # scalar version for gndr
    def log_prob_single(x):
        return -0.5 * jnp.sum((x @ prec) * x)
    # residual: log_prob = -0.5 ||L^T x||^2
    def residual_single(x):
        return L_prec.T @ x
    return log_prob, log_prob_single, residual_single, cov, prec


def make_rosenbrock(dim=10, a=1.0, b=100.0):
    """Rosenbrock.  x_even: mean=a, var=0.5.  x_odd: mean=1.5, var~2.505."""
    def log_prob(x):
        x_even = x[:, ::2]
        x_odd  = x[:, 1::2]
        return -(b * jnp.sum((x_odd - x_even**2)**2, axis=1)
                 + jnp.sum((x_even - a)**2, axis=1))
    def log_prob_single(x):
        x_even = x[::2]
        x_odd  = x[1::2]
        return -(b * jnp.sum((x_odd - x_even**2)**2)
                 + jnp.sum((x_even - a)**2))
    # GN residual: log_prob = -||r(x)||^2
    # r(x) = [sqrt(b)*(x_{i+1} - x_i^2), (x_i - a)]  for consecutive pairs
    sb = jnp.sqrt(b)
    def residual_single(x):
        x_even = x[::2]
        x_odd  = x[1::2]
        return jnp.concatenate([sb * (x_odd - x_even**2), x_even - a])
    return log_prob, log_prob_single, residual_single


# ══════════════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════════════

def report_gaussian(samples, true_cov):
    flat = samples.reshape(-1, samples.shape[-1])
    var_est  = jnp.var(flat, axis=0)
    var_true = jnp.diag(true_cov)
    rel_err  = jnp.mean(jnp.abs(var_est - var_true) / var_true)
    print(f"  rel_err(var)={rel_err:.3f}"
          f"  var=[{jnp.min(var_est):.2f}, {jnp.max(var_est):.2f}]"
          f"  (target: [{jnp.min(var_true):.2f}, {jnp.max(var_true):.2f}])")


def report_rosenbrock(samples, a=1.0):
    flat = samples.reshape(-1, samples.shape[-1])
    me = jnp.mean(flat[:, ::2]);  ve = jnp.mean(jnp.var(flat[:, ::2], axis=0))
    mo = jnp.mean(flat[:, 1::2]); vo = jnp.mean(jnp.var(flat[:, 1::2], axis=0))
    print(f"  x_even: mean={me:.3f} var={ve:.4f} (target: mean={a}, var=0.5)")
    print(f"  x_odd:  mean={mo:.3f} var={vo:.4f} (target: mean=1.5, var~2.505)")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    dim      = 10
    n_chains = 100
    n_samp   = 5000
    warmup   = 1000
    seed     = 123

    # ────────── Test 1: Gaussian ──────────
    print("\n" + "=" * 70)
    print(f"TEST 1: Ill-conditioned Gaussian  (D={dim}, kappa=100)")
    print("=" * 70)

    log_prob_g, log_prob_g_s, residual_g_s, cov_g, prec_g = make_gaussian(dim=dim, kappa=100.)
    init_g = jax.random.normal(jax.random.key(42), (n_chains, dim))

    print("\n--- Stretch ---")
    s, info = stretch.sampler_stretch(log_prob_g, init_g, n_samp, warmup=warmup, seed=seed)
    report_gaussian(s, cov_g)

    print("\n--- Side ---")
    s, info = side.sampler_side(log_prob_g, init_g, n_samp, warmup=warmup, seed=seed)
    report_gaussian(s, cov_g)

    print("\n--- Walk ---")
    s, info = walk.sampler_walk(log_prob_g, init_g, n_samp, warmup=warmup, seed=seed)
    report_gaussian(s, cov_g)

    print("\n--- DR Stretch ---")
    s, info = ensemble_dr.sampler_ensemble_dr_stretch(log_prob_g, init_g, n_samp, warmup=warmup, seed=seed)
    report_gaussian(s, cov_g)

    print("\n--- DR Side ---")
    s, info = ensemble_dr.sampler_ensemble_dr_side(log_prob_g, init_g, n_samp, warmup=warmup, seed=seed)
    report_gaussian(s, cov_g)

    print("\n--- Langevin Walk ---")
    s, info = langevin_walk.sampler_langevin_walk(log_prob_g, init_g, n_samp, warmup=warmup,
                                         step_size=0.01, seed=seed)
    report_gaussian(s, cov_g)

    print("\n--- Kalman (G=identity, M=precision) ---")
    forward_g = lambda x: x
    s, info = kalman_move.sampler_kalman_move(log_prob_g, forward_g, prec_g, init_g, n_samp,
                                    warmup=warmup, step_size=0.01, seed=seed)
    report_gaussian(s, cov_g)

    print("\n--- GN-DR (3-stage) ---")
    init_gndr = jax.random.normal(jax.random.key(42), (20, dim))
    s, info = gndr.sampler_gndr(log_prob_g_s, init_gndr, n_samp, warmup=warmup,
                                 step_size=0.5, n_try=3,
                                 residual_fn=residual_g_s, seed=seed)
    report_gaussian(s, cov_g)

    # ────────── Test 2: Rosenbrock ──────────
    print("\n" + "=" * 70)
    print(f"TEST 2: Rosenbrock  (D={dim}, a=1, b=100)")
    print("=" * 70)

    log_prob_r, log_prob_r_s, residual_r_s = make_rosenbrock(dim=dim)
    init_r = jax.random.normal(jax.random.key(42), (n_chains, dim))

    print("\n--- Stretch ---")
    s, info = stretch.sampler_stretch(log_prob_r, init_r, n_samp, warmup=warmup, seed=seed)
    report_rosenbrock(s)

    print("\n--- Side ---")
    s, info = side.sampler_side(log_prob_r, init_r, n_samp, warmup=warmup, seed=seed)
    report_rosenbrock(s)

    print("\n--- Walk ---")
    s, info = walk.sampler_walk(log_prob_r, init_r, n_samp, warmup=warmup,
                                step_size=0.1, seed=seed)
    report_rosenbrock(s)

    print("\n--- DR Stretch ---")
    s, info = ensemble_dr.sampler_ensemble_dr_stretch(log_prob_r, init_r, n_samp, warmup=warmup, seed=seed)
    report_rosenbrock(s)

    print("\n--- DR Side ---")
    s, info = ensemble_dr.sampler_ensemble_dr_side(log_prob_r, init_r, n_samp, warmup=warmup, seed=seed)
    report_rosenbrock(s)

    print("\n--- Langevin Walk ---")
    s, info = langevin_walk.sampler_langevin_walk(log_prob_r, init_r, n_samp, warmup=warmup,
                                         step_size=0.001, seed=seed)
    report_rosenbrock(s)

    print("\n--- GN-DR (3-stage) ---")
    init_gndr_r = jax.random.normal(jax.random.key(42), (20, dim))
    s, info = gndr.sampler_gndr(log_prob_r_s, init_gndr_r, n_samp, warmup=warmup,
                                 step_size=0.01, n_try=3,
                                 residual_fn=residual_r_s, seed=seed)
    report_rosenbrock(s)

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)