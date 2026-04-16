"""
example_funnel.py — Neal's funnel benchmark.

Target  (D = 1 + d, here d = D - 1):
    v ~ N(0, 9)
    x_i | v ~ N(0, exp(v))    i = 1, ..., d

so  log pi(v, x) = -v^2/18 - d·v/2 - 0.5·exp(-v)·||x||^2 + const.

The tight neck at v ≪ 0 makes this a classical stress-test for MCMC.

Exact moments:
    E[v] = 0, Var[v] = 9
    E[x_i] = 0, Var[x_i] = E[exp(v)] = exp(9/2) ≈ 90.02

Samplers compared:
    * sampler_stretch                — Goodman–Weare stretch
    * sampler_ensemble_dr_stretch    — 2-stage delayed-rejection stretch
    * sampler_gndr                   — Gauss–Newton proposal Langevin + DR
"""

from __future__ import annotations

import time
import jax
import jax.numpy as jnp
import numpy as np

from affine_invariant_samplers import (
    sampler_stretch,
    sampler_ensemble_dr_stretch,
    sampler_gndr,
    effective_sample_size,
)


# ──────────────────────────────────────────────────────────────────────────────
# Target:  Neal's funnel
# ──────────────────────────────────────────────────────────────────────────────

def make_funnel(dim=5):
    """Build log_prob (batched), log_prob_single, residual_fn (for gndr)."""
    d = dim - 1   # number of x-dimensions

    def log_prob(x):                                   # (batch, D) -> (batch,)
        v  = x[:, 0]
        xs = x[:, 1:]
        log_p_v = -0.5 * v ** 2 / 9.0
        log_p_x = (-0.5 * jnp.sum(xs ** 2 * jnp.exp(-v)[:, None], axis=1)
                   - 0.5 * d * v)
        return log_p_v + log_p_x

    def log_prob_single(z):                            # (D,) -> scalar
        v  = z[0]
        xs = z[1:]
        return (-0.5 * v ** 2 / 9.0
                - 0.5 * d * v
                - 0.5 * jnp.sum(xs ** 2) * jnp.exp(-v))

    def residual(z):                                   # (D,) -> (D,)
        # r = [v/3, exp(-v/2) * x_1, ..., exp(-v/2) * x_d]
        # ||r||^2 = v^2/9 + exp(-v)*||x||^2  => H_GN = J^T J captures
        # the v-dependent scaling of the x-components.
        v  = z[0]
        xs = z[1:]
        return jnp.concatenate([jnp.array([v / 3.0]),
                                jnp.exp(-v / 2.0) * xs])

    return log_prob, log_prob_single, residual


# ──────────────────────────────────────────────────────────────────────────────
# Report helper
# ──────────────────────────────────────────────────────────────────────────────

def _report(name, samples, accept, elapsed):
    flat = np.asarray(samples.reshape(-1, samples.shape[-1]))
    v    = flat[:, 0]
    xs   = flat[:, 1:]
    mv   = float(np.mean(v))
    vv   = float(np.var(v))
    vxs  = float(np.mean(np.var(xs, axis=0)))
    ess  = effective_sample_size(samples)
    ess_v = float(ess[0])
    print(f"  {name:<22s}  v: mean={mv:6.3f} var={vv:6.3f}   "
          f"x_i var={vxs:7.2f}   accept={accept:5.3f}   "
          f"ESS(v)={ess_v:6.1f}   time={elapsed:5.1f}s")


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dim      = 5
    n_chains = 100
    n_samp   = 10_000
    warmup   = 2_000
    seed     = 42

    log_prob, log_prob_single, residual = make_funnel(dim=dim)
    init_ens = jax.random.normal(jax.random.key(99), (n_chains, dim))
    init_hmc = jax.random.normal(jax.random.key(99), (30, dim))   # fewer for gndr (DR is expensive)

    print(f"Neal's funnel  D={dim}  n_chains={n_chains}  n_samp={n_samp}  warmup={warmup}")
    print("=" * 100)

    t0 = time.time()
    s, info = sampler_stretch(log_prob, init_ens, n_samp, warmup=warmup,
                               seed=seed, verbose=False)
    _report("stretch",         s, info["acceptance_rate"], time.time() - t0)

    t0 = time.time()
    s, info = sampler_ensemble_dr_stretch(log_prob, init_ens, n_samp, warmup=warmup,
                                           seed=seed, verbose=False)
    _report("stretch-DR",      s, info["acceptance_rate"], time.time() - t0)

    t0 = time.time()
    s, info = sampler_gndr(log_prob_single, init_hmc, n_samp, warmup=warmup,
                            step_size=0.5, n_try=3, residual_fn=residual,
                            seed=seed, verbose=False)
    _report("gndr",            s, info["acceptance_rate"], time.time() - t0)

    print("=" * 100)
    print("Target:  v mean=0, var=9   x_i var = exp(9/2) ≈ 90.02")
