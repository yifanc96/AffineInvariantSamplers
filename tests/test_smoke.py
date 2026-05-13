"""Smoke tests for every main-package sampler.

Two targets:
  * 2D correlated Gaussian  (ρ = 0.5, true μ = 0, true Σ given)
  * 10D Rosenbrock          (a = 1, b = 100; x_even ~ N(1, 1/2))

Each sampler runs briefly (small num_samples / warmup) and is judged only on
basic sanity — finite samples of the right shape, mean/variance within loose
tolerance of the true target.  Failures here indicate a regression in the
sampler, not a formal statistical guarantee.

Run with:
    pytest tests/
or directly:
    python tests/test_smoke.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from affine_invariant_samplers import (
    # ensemble affine-invariant
    sampler_walk,
    sampler_stretch,
    sampler_side,
    sampler_ensemble_dr_stretch,
    sampler_ensemble_dr_side,
    # ensemble gradient-based
    sampler_langevin_walk,
    sampler_kalman_move,
    sampler_kalman_dr,
    sampler_gndr,
    sampler_gndr_full,
    # HMC-family
    sampler_malt,
    sampler_mams,
    sampler_nuts,
    # ensemble HMC / microcanonical / NUTS
    sampler_peaches,
    sampler_peams,
    sampler_peanuts,
    sampler_pickles,
    sampler_chees,
    # unadjusted Langevin
    sampler_aldi,
    sampler_pickles_unadjusted,
)


# ──────────────────────────────────────────────────────────────────────────────
# Targets
# ──────────────────────────────────────────────────────────────────────────────

def make_gaussian_2d():
    """2D correlated Gaussian, μ = 0, Σ = [[1, 0.5], [0.5, 1]]."""
    dim = 2
    cov = jnp.array([[1.0, 0.5], [0.5, 1.0]])
    prec = jnp.linalg.inv(cov)

    def log_prob_batched(x):        # (batch, D) -> (batch,)
        return -0.5 * jnp.sum((x @ prec) * x, axis=-1)

    def log_prob_single(x):         # (D,) -> scalar
        return -0.5 * jnp.dot(x, prec @ x)

    def forward_single(x):          # residual s.t. L = -log pi ≈ 0.5 ||r||^2
        return jax.scipy.linalg.cholesky(prec, lower=False) @ x

    return dim, cov, log_prob_batched, log_prob_single, forward_single


def make_rosenbrock_10d():
    """10D Rosenbrock, a = 1, b = 100.
    True means: x_even ≈ 1, x_odd ≈ 2.  Variances ≈ (0.5, 2.5).
    """
    dim = 10
    a, b = 1.0, 100.0

    def log_prob_batched(x):
        xe = x[:, ::2]
        xo = x[:, 1::2]
        return -(b * jnp.sum((xo - xe ** 2) ** 2, axis=1)
                 + jnp.sum((xe - a) ** 2, axis=1))

    def log_prob_single(x):
        xe = x[::2]
        xo = x[1::2]
        return -(b * jnp.sum((xo - xe ** 2) ** 2)
                 + jnp.sum((xe - a) ** 2))

    return dim, a, b, log_prob_batched, log_prob_single


# ──────────────────────────────────────────────────────────────────────────────
# Assertion helpers
# ──────────────────────────────────────────────────────────────────────────────

def _assert_basic(samples, shape_expected):
    arr = jnp.asarray(samples)
    assert arr.shape[-len(shape_expected):] == shape_expected, \
        f"bad shape {arr.shape}, expected trailing {shape_expected}"
    assert jnp.all(jnp.isfinite(arr)), "non-finite samples"


def _assert_gaussian_2d(samples, cov, tol_mean=0.3, tol_cov=0.5):
    flat = jnp.asarray(samples).reshape(-1, 2)
    m = jnp.mean(flat, axis=0)
    c = jnp.cov(flat, rowvar=False)
    assert float(jnp.max(jnp.abs(m))) < tol_mean, \
        f"mean err {float(jnp.max(jnp.abs(m))):.3f} > {tol_mean}"
    assert float(jnp.max(jnp.abs(c - cov))) < tol_cov, \
        f"cov err {float(jnp.max(jnp.abs(c - cov))):.3f} > {tol_cov}"


def _assert_rosenbrock(samples, a=1.0, tol_mean=1.0):
    flat = jnp.asarray(samples).reshape(-1, 10)
    me = float(jnp.mean(flat[:, ::2]))      # should be ~ a = 1
    mo = float(jnp.mean(flat[:, 1::2]))     # should be ~ 2 (a^2 + var)
    assert abs(me - a) < tol_mean, f"x_even mean {me:.2f} far from {a}"
    assert abs(mo - 2.0) < 2.0,     f"x_odd mean {mo:.2f} far from 2"


# ──────────────────────────────────────────────────────────────────────────────
# Tests — 2D Gaussian
# ──────────────────────────────────────────────────────────────────────────────

NUM_SAMPLES = 500
WARMUP      = 200
SEED        = 0


def test_gaussian_2d_all():
    dim, cov, lp, lp_s, fwd_s = make_gaussian_2d()
    init_ens = jax.random.normal(jax.random.key(42), (20, dim))
    init_hmc = jax.random.normal(jax.random.key(42), (10, dim))

    # ensemble affine-invariant (gradient-free, batched log_prob)
    for fn in (sampler_walk, sampler_stretch, sampler_side,
               sampler_ensemble_dr_stretch, sampler_ensemble_dr_side):
        s, info = fn(lp, init_ens, NUM_SAMPLES, warmup=WARMUP, seed=SEED, verbose=False)
        _assert_basic(s, (20, dim))
        _assert_gaussian_2d(s, cov)

    # ensemble gradient-based
    for fn in (sampler_langevin_walk,):
        s, info = fn(lp, init_ens, NUM_SAMPLES, warmup=WARMUP, seed=SEED, verbose=False)
        _assert_basic(s, (20, dim))
        _assert_gaussian_2d(s, cov)

    # Kalman samplers need forward_fn + M
    for fn in (sampler_kalman_move, sampler_kalman_dr):
        s, info = fn(lp, lambda x: x, jnp.linalg.inv(cov),
                     init_ens, NUM_SAMPLES, warmup=WARMUP, seed=SEED, verbose=False)
        _assert_basic(s, (20, dim))
        _assert_gaussian_2d(s, cov)

    # gndr (single-point log_prob, optional residual)
    s, info = sampler_gndr(lp_s, init_hmc, NUM_SAMPLES, warmup=WARMUP,
                           seed=SEED, verbose=False, residual_fn=fwd_s)
    _assert_basic(s, (10, dim))
    _assert_gaussian_2d(s, cov)

    # Single-chain HMC/MAMS (single-point log_prob)
    for fn in (sampler_malt, sampler_mams):
        s, info = fn(lp_s, init_hmc, NUM_SAMPLES, warmup=WARMUP, seed=SEED, verbose=False)
        _assert_basic(s, (10, dim))
        _assert_gaussian_2d(s, cov)

    # NUTS (batched log_prob)
    s, info = sampler_nuts(lp, init_hmc, NUM_SAMPLES, warmup=WARMUP,
                           seed=SEED, verbose=False)
    _assert_basic(s, (10, dim))
    _assert_gaussian_2d(s, cov)

    # ensemble HMC variants (batched log_prob)
    for fn in (sampler_peaches, sampler_peams, sampler_peanuts,
               sampler_pickles, sampler_chees):
        s, info = fn(lp, init_ens, NUM_SAMPLES, warmup=WARMUP, seed=SEED, verbose=False)
        _assert_basic(s, (20, dim))
        _assert_gaussian_2d(s, cov)

    # unadjusted Langevin (aldi, pickles_unadjusted) — looser bounds (no MH)
    for fn in (sampler_aldi, sampler_pickles_unadjusted):
        s, info = fn(lp, init_ens, NUM_SAMPLES, warmup=WARMUP, seed=SEED, verbose=False)
        _assert_basic(s, (20, dim))
        _assert_gaussian_2d(s, cov, tol_mean=0.5, tol_cov=0.8)


# ──────────────────────────────────────────────────────────────────────────────
# Tests — 10D Rosenbrock
# ──────────────────────────────────────────────────────────────────────────────

NUM_SAMPLES_R = 1000
WARMUP_R      = 500


def test_rosenbrock_10d_all():
    dim, a, b, lp, lp_s = make_rosenbrock_10d()
    init_ens = jax.random.normal(jax.random.key(42), (40, dim))
    init_hmc = jax.random.normal(jax.random.key(42), (20, dim))

    # ensemble affine-invariant
    for fn in (sampler_walk, sampler_stretch, sampler_side,
               sampler_ensemble_dr_stretch, sampler_ensemble_dr_side):
        s, info = fn(lp, init_ens, NUM_SAMPLES_R, warmup=WARMUP_R,
                     seed=SEED, verbose=False)
        _assert_basic(s, (40, dim))
        _assert_rosenbrock(s, a=a, tol_mean=1.5)

    # Langevin walk
    s, info = sampler_langevin_walk(lp, init_ens, NUM_SAMPLES_R,
                                     warmup=WARMUP_R, seed=SEED, verbose=False)
    _assert_basic(s, (40, dim))
    _assert_rosenbrock(s, a=a, tol_mean=1.5)

    # Single-chain HMC/MAMS (single-point log_prob)
    for fn in (sampler_malt, sampler_mams):
        s, info = fn(lp_s, init_hmc, NUM_SAMPLES_R, warmup=WARMUP_R,
                     seed=SEED, verbose=False)
        _assert_basic(s, (20, dim))
        _assert_rosenbrock(s, a=a, tol_mean=1.5)

    # NUTS (batched log_prob)
    s, info = sampler_nuts(lp, init_hmc, NUM_SAMPLES_R, warmup=WARMUP_R,
                           seed=SEED, verbose=False)
    _assert_basic(s, (20, dim))
    _assert_rosenbrock(s, a=a, tol_mean=1.5)

    # ensemble HMC / NUTS / ChEES
    for fn in (sampler_peaches, sampler_peams, sampler_peanuts,
               sampler_pickles, sampler_chees):
        s, info = fn(lp, init_ens, NUM_SAMPLES_R, warmup=WARMUP_R,
                     seed=SEED, verbose=False)
        _assert_basic(s, (40, dim))
        _assert_rosenbrock(s, a=a, tol_mean=1.5)


# ──────────────────────────────────────────────────────────────────────────────
# Run directly
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# DR-for-MALA on light-tailed |x|^4 — gndr_full with identity preconditioner.
# Vanilla MALA loses geometric ergodicity on exp(-|x|^4 / 4); multi-stage DR
# is expected to restore it.  Setting hessian_fn = I removes the GN structure
# from gndr_full so the proposal is just MALA, and only DR varies.
# ──────────────────────────────────────────────────────────────────────────────

# True var per dim of exp(-x^4 / 4):  4^{1/2} · Γ(3/4) / Γ(1/4)  ≈  0.6760
_VAR_QUARTIC_TRUE = 0.6760


def make_quartic(dim=3):
    """log π(x) = -||x||^4 / 4  (i.i.d. light-tailed components, p = 4)."""
    def lp_b(x):
        return -0.25 * jnp.sum(x ** 4, axis=-1)
    def lp_s(x):
        return -0.25 * jnp.sum(x ** 4)
    return dim, lp_b, lp_s


def _assert_quartic(samples, dim, tol=0.20):
    flat = jnp.asarray(samples).reshape(-1, dim)
    m = float(jnp.max(jnp.abs(jnp.mean(flat, axis=0))))
    v = float(jnp.mean(jnp.var(flat, axis=0)))
    assert m < 0.3, f"per-dim mean drift {m:.3f}, target 0"
    rel = abs(v - _VAR_QUARTIC_TRUE) / _VAR_QUARTIC_TRUE
    assert rel < tol, f"per-dim var {v:.4f} off from {_VAR_QUARTIC_TRUE} by {rel*100:.1f}%"


def test_quartic_dr_mala():
    """gndr_full with H = I on |x|^4 / 4, init in the tail.

    Identity preconditioner means the proposal is pure MALA — the only
    moving part is the DR ladder.  At n_try=1 this is vanilla MALA, which
    is known to lose geometric ergodicity on |x|^p with p > 2.  At n_try=5
    the DR retries with shrunk step sizes rescue the chain.

    Test:  with n_try=5 and init far in the tail (≈ 3.5σ out), the chain
    reaches the bulk and recovers the variance to within 20%.
    """
    dim, lp_b, lp_s = make_quartic(dim=3)
    n_chains = 32
    init = 3.0 + 0.3 * jax.random.normal(jax.random.key(SEED), (n_chains, dim))

    H_identity = lambda x: jnp.eye(dim)
    s, info = sampler_gndr_full(
        lp_s, init, num_samples=1500, warmup=500,
        step_size=0.1, n_try=5, hessian_fn=H_identity,
        seed=SEED, verbose=False, find_init_step_size=False,
    )
    _assert_basic(s, (n_chains, dim))
    _assert_quartic(s, dim, tol=0.20)


if __name__ == "__main__":
    print("2D Gaussian smoke tests …", flush=True)
    test_gaussian_2d_all()
    print("  OK")
    print("10D Rosenbrock smoke tests …", flush=True)
    test_rosenbrock_10d_all()
    print("  OK")
    print("DR-for-MALA on |x|^4 …", flush=True)
    test_quartic_dr_mala()
    print("  OK")
    print("\nAll smoke tests passed.")
