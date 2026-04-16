"""
run_samplers.py — Import and run all four samplers on test distributions.

Samplers
--------
All samplers share the same interface:

    samples, info = sampler_xxx(log_prob_fn, initial_state, num_samples, **kwargs)

    log_prob_fn    : callable, (n_chains, D) -> (n_chains,).  Batched log density.
    initial_state  : array (n_chains, D).  Starting positions for all chains.
    num_samples    : int.  Number of post-warmup samples to collect.

    Returns:
        samples    : array (num_samples, n_chains, D)
        info       : dict with diagnostics

Sampler overview
----------------
  sampler_nuts     Standard NUTS.  Single-chain dynamics, identity mass matrix.
                   Best baseline — no ensemble, no preconditioning.
                   Adapts: step size (dual averaging).
                   Trajectory length: NUTS tree (automatic).

  sampler_chess    Standard ChEES HMC.  Single-chain, identity mass matrix.
                   Adapts: step size (dual averaging) + trajectory length (ChEES).
                   Production uses Halton-jittered L to avoid resonances.

  sampler_peaches  Ensemble-preconditioned ChEES HMC (PEACHES).
                   Uses complement ensemble for preconditioning (h-walk or h-side).
                   Adapts: step size (DA) + trajectory length (ChEES).
                   Requires n_chains >= 4, even.

  sampler_peanuts  Ensemble-preconditioned NUTS (PEANUTS).
                   Uses complement ensemble for preconditioning (h-walk or h-side).
                   Adapts: step size (DA).  Trajectory length: NUTS tree (automatic).
                   Requires n_chains >= 4, even.
                   Options: uturn="affine-invariant" (default) or "euclidean".

Key parameters (shared)
-----------------------
  warmup         : int, warmup iterations for adaptation (default 1000)
  step_size      : float, initial step size (adapted during warmup)
  target_accept  : float, target acceptance rate for dual averaging (default 0.651)
  thin_by        : int, thinning factor (default 1)
  seed           : int, random seed
  verbose        : bool, print progress (default True)

Ensemble-specific (peaches, peanuts)
------------------------------------
  move           : "h-walk" (default) or "h-side"
                   h-walk: momentum in the span of complement ensemble (D-dim)
                   h-side: scalar momentum along a random ensemble direction (1-dim)

NUTS-specific (nuts, peanuts)
-----------------------------
  sampling       : "progressive" (default, biased) or "multinomial" (unbiased)
  max_tree_depth : int, max NUTS tree depth (default 10, trajectory <= 2^depth)

ChEES-specific (chess, peaches)
-------------------------------
  L              : int, initial leapfrog steps (adapted during warmup)
  max_L          : int, maximum leapfrog steps (default 100)

Usage
-----
    import jax
    import jax.numpy as jnp
    from affine_invariant_samplers import sampler_nuts

    # Define a batched log density: (n_chains, D) -> (n_chains,)
    def log_prob(x):
        return -0.5 * jnp.sum(x**2, axis=-1)

    init = jax.random.normal(jax.random.key(0), (20, 5))
    samples, info = sampler_nuts(log_prob, init, num_samples=2000)
"""

import jax
import jax.numpy as jnp

from affine_invariant_samplers import nuts as sampler_nuts
from affine_invariant_samplers import chess as sampler_chess
from affine_invariant_samplers import peaches as sampler_peaches
from affine_invariant_samplers import peanuts as sampler_peanuts


# ══════════════════════════════════════════════════════════════════════════════
# Test distributions
# ══════════════════════════════════════════════════════════════════════════════

def make_gaussian(dim=20, kappa=1000., seed=0):
    """Ill-conditioned Gaussian.  Returns (log_prob, log_prob_single, forward_single, true_cov).

    forward_single: (D,) -> (D,)  such that Phi(x) = 0.5*||F(x)||^2 = 0.5*x^T Prec x.
    """
    eigvals = jnp.logspace(0, jnp.log10(kappa), dim)
    Q, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(seed), (dim, dim)))
    cov  = Q @ jnp.diag(eigvals) @ Q.T
    prec = Q @ jnp.diag(1. / eigvals) @ Q.T
    L_prec = jnp.linalg.cholesky(prec)       # Prec = L L^T
    def log_prob(x):
        return -0.5 * jnp.sum((x @ prec) * x, axis=-1)
    def log_prob_single(x):
        return -0.5 * jnp.sum((x @ prec) * x)
    def forward_single(x):
        return L_prec.T @ x                   # ||L^T x||^2 = x^T L L^T x = x^T Prec x
    return log_prob, log_prob_single, forward_single, cov


def make_rosenbrock(dim=20, a=1.0, b=100.0):
    """Rosenbrock distribution in dim dimensions (dim must be even).
    Exact moments:
        x_even: mean=a, var=0.5
        x_odd:  mean=a^2+0.5=1.5, var~2.505

    forward_single: (D,) -> (D,)  residual form so Phi(x) = 0.5*||F(x)||^2.
    """
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
    def forward_single(x):
        x_even = x[::2]
        x_odd  = x[1::2]
        return jnp.concatenate([
            jnp.sqrt(2.0 * b) * (x_odd - x_even**2),
            jnp.sqrt(2.0) * (x_even - a),
        ])
    return log_prob, log_prob_single, forward_single


def make_funnel(dim=3):
    """Neal's funnel.  v ~ N(0,9), x_i|v ~ N(0, exp(v)).
    Exact: E[v]=0, Var[v]=9, E[x_i]=0, Var[x_i]=exp(9/2)~90.0
    """
    def log_prob(x):
        v  = x[:, 0]
        xs = x[:, 1:]
        log_p_v = -0.5 * v**2 / 9.
        log_p_x = (-0.5 * jnp.sum(xs**2 * jnp.exp(-v)[:, None], axis=1)
                   - 0.5 * (dim - 1) * v)
        return log_p_v + log_p_x
    return log_prob


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════��═══════════════════════════

def report_gaussian(samples, true_cov, label=""):
    flat = samples.reshape(-1, samples.shape[-1])
    var_est  = jnp.var(flat, axis=0)
    var_true = jnp.diag(true_cov)
    rel_err  = jnp.mean(jnp.abs(var_est - var_true) / var_true)
    print(f"  {label}mean_rel_err(var)={rel_err:.3f}"
          f"  var_range=[{jnp.min(var_est):.2f}, {jnp.max(var_est):.2f}]"
          f"  (target: [{jnp.min(var_true):.2f}, {jnp.max(var_true):.2f}])")


def report_rosenbrock(samples, a=1.0, label=""):
    flat = samples.reshape(-1, samples.shape[-1])
    me = jnp.mean(flat[:, ::2]);  ve = jnp.mean(jnp.var(flat[:, ::2], axis=0))
    mo = jnp.mean(flat[:, 1::2]); vo = jnp.mean(jnp.var(flat[:, 1::2], axis=0))
    print(f"  {label}x_even: mean={me:.3f} var={ve:.4f} (target: mean={a}, var=0.5)")
    print(f"  {label}x_odd:  mean={mo:.3f} var={vo:.4f} (target: mean=1.5, var~2.505)")


def report_funnel(samples, label=""):
    flat = samples.reshape(-1, samples.shape[-1])
    v = flat[:, 0]; xs = flat[:, 1:]
    print(f"  {label}v:   mean={jnp.mean(v):.3f}  var={jnp.var(v):.2f}  (target: 0, 9)")
    print(f"  {label}x_i: mean={jnp.mean(xs):.3f}  var={jnp.mean(jnp.var(xs, axis=0)):.1f}  (target: 0, ~90)")


# ══════════════════════════════════════════════════════════════════════════════
# Run all samplers
# ════════════���═════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Shared settings
    dim         = 10
    n_chains    = 100          # must be even & >= 4 for ensemble samplers
    num_samples = 2000
    warmup      = 400
    seed        = 123

    # ---------- Test 1: Ill-conditioned Gaussian ----------
    print("\n" + "=" * 70)
    print("TEST 1: Ill-conditioned Gaussian  (D=20, kappa=1000)")
    print("=" * 70)

    log_prob_g, log_prob_g_single, forward_g, cov_g = make_gaussian(dim=dim, kappa=1000.)
    init_g = jax.random.normal(jax.random.key(42), (n_chains, dim))

    print("\n--- NUTS ---")
    s, info = sampler_nuts.sampler_nuts(
        log_prob_g, init_g, num_samples, warmup=warmup, seed=seed)
    report_gaussian(s, cov_g)
    print(f"  info: {info}")

    print("\n--- ChEES ---")
    s, info = sampler_chess.sampler_chess(
        log_prob_g, init_g, num_samples, warmup=warmup, seed=seed)
    report_gaussian(s, cov_g)
    print(f"  info: {info}")

    print("\n--- PEACHES (affine-invariant) ---")
    s, info = sampler_peaches.sampler_peaches(
        log_prob_g, init_g, num_samples, warmup=warmup, seed=seed,
        step_size=0.01, chees_metric="affine-invariant")
    report_gaussian(s, cov_g)
    print(f"  info: {info}")

    print("\n--- PEANUTS (affine-invariant) ---")
    s, info = sampler_peanuts.sampler_peanuts(
        log_prob_g, init_g, num_samples, warmup=warmup, seed=seed,
        step_size=0.01, uturn="affine-invariant")
    report_gaussian(s, cov_g)
    print(f"  info: {info}")

    # ---------- Test 2: Rosenbrock ----------
    print("\n" + "=" * 70)
    print("TEST 2: Rosenbrock  (D=20, a=1, b=100)")
    print("=" * 70)

    log_prob_r, log_prob_r_single, forward_r = make_rosenbrock(dim=dim)
    init_r = jax.random.normal(jax.random.key(42), (n_chains, dim))

    print("\n--- NUTS ---")
    s, info = sampler_nuts.sampler_nuts(
        log_prob_r, init_r, num_samples, warmup=warmup, seed=seed, step_size=0.01)
    report_rosenbrock(s)
    print(f"  info: {info}")

    print("\n--- ChEES ---")
    s, info = sampler_chess.sampler_chess(
        log_prob_r, init_r, num_samples, warmup=warmup, seed=seed, step_size=0.01)
    report_rosenbrock(s)
    print(f"  info: {info}")

    print("\n--- PEACHES ---")
    s, info = sampler_peaches.sampler_peaches(
        log_prob_r, init_r, num_samples, warmup=warmup, seed=seed, step_size=0.01)
    report_rosenbrock(s)
    print(f"  info: {info}")

    print("\n--- PEANUTS ---")
    s, info = sampler_peanuts.sampler_peanuts(
        log_prob_r, init_r, num_samples, warmup=warmup, seed=seed, step_size=0.01)
    report_rosenbrock(s)
    print(f"  info: {info}")

    # ---------- Test 3: Neal's Funnel ----------
    # print("\n" + "=" * 70)
    # print("TEST 3: Neal's Funnel  (D=3)")
    # print("=" * 70)

    # funnel_dim  = 3
    # log_prob_f  = make_funnel(dim=funnel_dim)
    # init_f = jax.random.normal(jax.random.key(99), (n_chains, funnel_dim)) * 0.5

    # print("\n--- NUTS ---")
    # s, info = sampler_nuts.sampler_nuts(
    #     log_prob_f, init_f, num_samples, warmup=warmup, seed=seed)
    # report_funnel(s)
    # print(f"  info: {info}")

    # print("\n--- ChEES ---")
    # s, info = sampler_chess.sampler_chess(
    #     log_prob_f, init_f, num_samples, warmup=warmup, seed=seed)
    # report_funnel(s)
    # print(f"  info: {info}")

    # print("\n--- PEACHES ---")
    # s, info = sampler_peaches.sampler_peaches(
    #     log_prob_f, init_f, num_samples, warmup=warmup, seed=seed, step_size=0.01)
    # report_funnel(s)
    # print(f"  info: {info}")

    # print("\n--- PEANUTS ---")
    # s, info = sampler_peanuts.sampler_peanuts(
    #     log_prob_f, init_f, num_samples, warmup=warmup, seed=seed)
    # report_funnel(s)
    # print(f"  info: {info}")

    # print("\n" + "=" * 70)
    # print("Done.")
    # print("=" * 70)