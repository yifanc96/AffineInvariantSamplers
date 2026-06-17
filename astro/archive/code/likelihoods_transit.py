"""jaxoplanet single-planet transit benchmark.

Fits a synthetic Kepler/TESS-like transit light curve with a quadratic
limb-darkened transit model from `jaxoplanet`.  The posterior is 7-D with
strong nonlinear correlations (the canonical ``(period, t0, R_p/R_*, b,
u1, u2, log σ)`` banana), which is the regime where gradient-based
samplers are designed to shine.

For reproducibility we use a synthetic light curve generated with fixed
parameters + Gaussian noise — no external Kepler FITS dependency.  To
swap in a real TESS target, replace the data block in ``make_transit`` with
``lightkurve`` / ``exoplanet`` data (both give `(t, flux, flux_err)`);
the likelihood code is unchanged.

The emcee / `exoplanet` tutorial "RV with JAX" and the `jaxoplanet`
transit tutorial use the same parameterisation, so the benchmark is
directly comparable to published numbers.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxoplanet.orbits import keplerian
from jaxoplanet.light_curves import limb_dark_light_curve


# True parameters for the synthetic target (Kepler-10b-ish: short period,
# small planet, bright star).
_TRUTH = dict(period=3.2, t0=0.1, r=0.05, b=0.4, u1=0.4, u2=0.25,
              log_sigma=np.log(200e-6))


def _make_synthetic(seed: int = 0, n_obs: int = 3000, t_span: float = 12.0):
    """Generate a noisy transit light curve around the truth."""
    t = np.linspace(0.0, t_span, n_obs)
    system = keplerian.System(central=keplerian.Central(mass=1.0, radius=1.0))
    system = system.add_body(
        period=_TRUTH["period"], time_transit=_TRUTH["t0"],
        radius=_TRUTH["r"], impact_param=_TRUTH["b"],
        eccentricity=0.0, omega_peri=0.0,
    )
    model = np.asarray(limb_dark_light_curve(system, [_TRUTH["u1"], _TRUTH["u2"]])(t))[:, 0]
    rng = np.random.default_rng(seed)
    sigma = np.exp(_TRUTH["log_sigma"])
    flux = 1.0 + model + sigma * rng.standard_normal(n_obs)
    return t, flux, np.full(n_obs, sigma)


def make_transit(seed: int = 0, n_obs: int = 3000):
    """Single-planet transit fit — 7-D log posterior.

    θ = (period, t0, r, b, u1, u2, log_sigma) with uniform priors:

        period     ∈ [2.5, 4.0]        (days)
        t0         ∈ [-0.5, 1.0]       (days)
        r = Rp/R*  ∈ [0.01, 0.15]
        b          ∈ [0.0, 1.0]        (impact parameter)
        u1, u2     ∈ [0.0, 1.0]        (quadratic limb-darkening, with
                                        physicality constraint u1+u2<1,
                                        u1>0, u1+2u2>0 via re-parameterisation)
        log_sigma  ∈ [-10, -5]         (white-noise jitter)
    """
    t_np, flux_np, _ = _make_synthetic(seed=seed, n_obs=n_obs)
    t    = jnp.asarray(t_np)
    flux = jnp.asarray(flux_np)

    LO = jnp.asarray([2.5, -0.5, 0.01, 0.0, 0.0, 0.0, -10.0])
    HI = jnp.asarray([4.0,  1.0, 0.15, 1.0, 1.0, 1.0,  -5.0])

    def _lc(theta):
        period, t0, r, b, u1, u2, _ = theta
        system = keplerian.System(central=keplerian.Central(mass=1.0, radius=1.0))
        system = system.add_body(period=period, time_transit=t0, radius=r,
                                 impact_param=b, eccentricity=0.0, omega_peri=0.0)
        return limb_dark_light_curve(system, jnp.stack([u1, u2]))(t)[:, 0]

    def _one_lp(theta):
        lc = _lc(theta)
        model = 1.0 + lc
        sigma = jnp.exp(theta[6])
        resid = (flux - model) / sigma
        return -0.5 * jnp.sum(resid ** 2) - len(t) * (theta[6] + 0.5 * jnp.log(2 * jnp.pi))

    def log_prob(theta):
        # (B, 7) -> (B,)
        in_box = jnp.all((theta >= LO) & (theta <= HI), axis=1)
        # Additional physicality: u1 > 0, u1 + u2 < 1, u1 + 2 u2 > 0.
        u1, u2 = theta[:, 4], theta[:, 5]
        ld_ok  = (u1 > 0) & (u1 + u2 < 1) & (u1 + 2 * u2 > 0)
        valid  = in_box & ld_ok
        lp     = jax.vmap(_one_lp)(theta)
        return jnp.where(valid, lp, -jnp.inf)

    info = dict(dim=7, n_data=n_obs,
                param_names=["period", "t0", "r", "b", "u1", "u2", "log_sigma"],
                prior_lo=np.asarray(LO), prior_hi=np.asarray(HI),
                truth={k: float(v) for k, v in _TRUTH.items()})
    return log_prob, info


def init_transit(key, n_chains: int):
    """Small Gaussian ball around the truth (for gradient samplers).
    For emcee / stretch, this is also the standard warm-start pattern."""
    mean  = jnp.asarray([_TRUTH["period"], _TRUTH["t0"], _TRUTH["r"],
                         _TRUTH["b"], _TRUTH["u1"], _TRUTH["u2"],
                         _TRUTH["log_sigma"]])
    scale = jnp.asarray([0.01, 0.01, 0.003, 0.05, 0.05, 0.05, 0.1])
    return mean + scale * jax.random.normal(key, (n_chains, 7))
