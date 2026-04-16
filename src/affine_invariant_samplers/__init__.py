"""
affine_invariant_samplers
=========================

Affine-invariant ensemble MCMC samplers and their dimensional scaling.

Reference: Chen, "New affine invariant ensemble samplers and their dimensional
scaling", arXiv:2505.02987.

Quick start
-----------
>>> import jax, jax.numpy as jnp
>>> from affine_invariant_samplers import sampler_walk
>>> def log_prob(x):                                  # (batch, D) -> (batch,)
...     return -0.5 * jnp.sum(x * x, axis=-1)
>>> init = jax.random.normal(jax.random.key(0), (20, 2))
>>> samples, info = sampler_walk(log_prob, init, num_samples=1000, warmup=500)

Samplers are also accessible namespaced by module, e.g.
``from affine_invariant_samplers.walk import sampler_walk``.
"""

# ─── Ensemble affine-invariant (gradient-free, MH-adjusted) ──────────────────
from .walk        import sampler_walk
from .stretch     import sampler_stretch
from .side        import sampler_side
from .ensemble_dr import sampler_ensemble_dr_stretch, sampler_ensemble_dr_side

# ─── Ensemble gradient-based ────────────────────────────────────────────────
from .langevin_walk import sampler_langevin_walk
from .kalman_move   import sampler_kalman_move
from .kalman_dr     import sampler_kalman_dr
from .gndr          import sampler_gndr

# ─── HMC-family (single chain, batched) ─────────────────────────────────────
from .malt import sampler_malt
from .mams import sampler_mams
from .nuts import sampler_nuts

# ─── Ensemble HMC / microcanonical / NUTS ───────────────────────────────────
from .peaches import sampler_peaches
from .peams   import sampler_peams
from .peanuts import sampler_peanuts
from .pickles import sampler_pickles
from .chess   import sampler_chess

# ─── Unadjusted Langevin dynamics (ensemble / interacting) ──────────────────
from .aldi               import sampler_aldi
from .pickles_unadjusted import sampler_pickles_unadjusted


__all__ = [
    # ensemble affine-invariant
    "sampler_walk",
    "sampler_stretch",
    "sampler_side",
    "sampler_ensemble_dr_stretch",
    "sampler_ensemble_dr_side",
    # ensemble gradient-based
    "sampler_langevin_walk",
    "sampler_kalman_move",
    "sampler_kalman_dr",
    "sampler_gndr",
    # HMC-family
    "sampler_malt",
    "sampler_mams",
    "sampler_nuts",
    # ensemble HMC / microcanonical / NUTS
    "sampler_peaches",
    "sampler_peams",
    "sampler_peanuts",
    "sampler_pickles",
    "sampler_chess",
    # unadjusted Langevin
    "sampler_aldi",
    "sampler_pickles_unadjusted",
]

__version__ = "0.1.0"
