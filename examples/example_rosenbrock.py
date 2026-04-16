"""
example_rosenbrock.py — 10-D Rosenbrock benchmark.

Target:
    log pi(x) = -( b · sum_i (x_{2i+1} - x_{2i}^2)^2  +  sum_i (x_{2i} - a)^2 )
    with (a, b) = (1, 100) and D = 10  (so 5 (even, odd) pairs).

Exact marginal moments:
    x_even ~ N(a, 1/2)                   mean = a = 1, var = 0.5
    x_odd | x_even ~ N(x_even^2, 1/(2b))
      => E[x_odd]   = E[x_even^2] = a^2 + 0.5 = 1.5
         Var[x_odd] = Var(x_even^2) + 1/(2b) ≈ 2.505

Samplers compared:
    * sampler_peaches  — ensemble-preconditioned HMC (walk + HMC)
    * sampler_pickles  — parallel interacting kinetic Langevin
    * sampler_peams    — ensemble-preconditioned microcanonical HMC
"""

from __future__ import annotations

import time
import jax
import jax.numpy as jnp

from affine_invariant_samplers import (
    sampler_peaches,
    sampler_pickles,
    sampler_peams,
    effective_sample_size,
)


# ──────────────────────────────────────────────────────────────────────────────
# Target:  Rosenbrock in D dims  (D must be even)
# ──────────────────────────────────────────────────────────────────────────────

def make_rosenbrock(dim=10, a=1.0, b=100.0):
    assert dim % 2 == 0, "dim must be even"

    def log_prob(x):                          # (batch, D) -> (batch,)
        xe = x[:, ::2]
        xo = x[:, 1::2]
        return -(b * jnp.sum((xo - xe ** 2) ** 2, axis=1)
                 + jnp.sum((xe - a) ** 2, axis=1))

    return log_prob


# ──────────────────────────────────────────────────────────────────────────────
# Report helper
# ──────────────────────────────────────────────────────────────────────────────

def _report(name, samples, a, accept, elapsed):
    flat = jnp.asarray(samples).reshape(-1, samples.shape[-1])
    xe, xo  = flat[:, ::2], flat[:, 1::2]
    me      = float(jnp.mean(xe));                ve = float(jnp.mean(jnp.var(xe, axis=0)))
    mo      = float(jnp.mean(xo));                vo = float(jnp.mean(jnp.var(xo, axis=0)))
    ess     = effective_sample_size(samples)
    print(f"  {name:<20s}  x_e mean={me:5.2f} var={ve:5.2f}   "
          f"x_o mean={mo:5.2f} var={vo:5.2f}   "
          f"accept={accept:5.3f}   min_ESS={float(ess.min()):7.1f}   "
          f"time={elapsed:5.1f}s")


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dim      = 10
    a, b     = 1.0, 100.0
    n_chains = 100
    n_samp   = 5000
    warmup   = 1000
    seed     = 123

    log_prob = make_rosenbrock(dim=dim, a=a, b=b)
    init = jax.random.normal(jax.random.key(42), (n_chains, dim))

    print(f"Rosenbrock  D={dim}  (a, b) = ({a}, {b})  "
          f"n_chains={n_chains}  n_samp={n_samp}  warmup={warmup}")
    print("=" * 100)

    t0 = time.time()
    s, info = sampler_peaches(log_prob, init, n_samp, warmup=warmup,
                               step_size=0.01, seed=seed, verbose=False)
    _report("peaches", s, a, info["acceptance_rate"], time.time() - t0)

    t0 = time.time()
    s, info = sampler_pickles(log_prob, init, n_samp, warmup=warmup,
                               step_size=0.01, gamma=2.0, seed=seed, verbose=False)
    _report("pickles", s, a, info["acceptance_rate"], time.time() - t0)

    t0 = time.time()
    s, info = sampler_peams(log_prob, init, n_samp, warmup=warmup,
                             step_size=0.01, seed=seed, verbose=False)
    _report("peams",   s, a, info["acceptance_rate"], time.time() - t0)

    print("=" * 100)
    print(f"Target: x_e mean={a}, var=0.50   x_o mean={a**2 + 0.5:.2f}, var≈2.505")
