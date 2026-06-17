"""3-planet transit fit with jaxoplanet — 15-D benchmark.

θ = (P_1, t0_1, r_1, b_1,
     P_2, t0_2, r_2, b_2,
     P_3, t0_3, r_3, b_3,
     u1, u2, log_sigma)                                 (15 dims)

Synthetic Kepler-36-like triple system (nearly-resonant periods → strong
transit-timing correlations between planets).  This puts the problem in
the D≥10 regime where gradient-based samplers are expected to dominate,
and the shared limb-darkening (u1, u2) creates a known degeneracy across
all 3 planets' transit depths.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxoplanet.orbits import keplerian
from jaxoplanet.light_curves import limb_dark_light_curve


# Near-resonant: P_2/P_1 ≈ 1.35, P_3/P_1 ≈ 1.62 (Kepler-36-ish but shifted
# to avoid exact commensurability).  Different depths so all three are
# individually identifiable.
_TRUTH = dict(
    P=[3.20,  4.31,  5.19],
    t0=[0.10, 0.55, 1.20],
    r=[0.050, 0.042, 0.035],
    b=[0.40,  0.25,  0.55],
    u1=0.40, u2=0.25,
    log_sigma=np.log(200e-6),
)


def _make_synthetic(seed: int = 0, n_obs: int = 5000, t_span: float = 30.0):
    """Generate noisy light curve with 3 transiting planets."""
    t = np.linspace(0.0, t_span, n_obs)
    system = keplerian.System(central=keplerian.Central(mass=1.0, radius=1.0))
    for i in range(3):
        system = system.add_body(period=_TRUTH["P"][i],
                                  time_transit=_TRUTH["t0"][i],
                                  radius=_TRUTH["r"][i],
                                  impact_param=_TRUTH["b"][i],
                                  eccentricity=0.0, omega_peri=0.0)
    lc = np.asarray(limb_dark_light_curve(system, [_TRUTH["u1"], _TRUTH["u2"]])(t))
    model = 1.0 + lc.sum(axis=1)
    sigma = np.exp(_TRUTH["log_sigma"])
    rng = np.random.default_rng(seed)
    flux = model + sigma * rng.standard_normal(n_obs)
    return t, flux


def make_multi_transit(seed: int = 0, n_obs: int = 5000):
    t_np, flux_np = _make_synthetic(seed=seed, n_obs=n_obs)
    t    = jnp.asarray(t_np)
    flux = jnp.asarray(flux_np)

    # param order: [P_1, t0_1, r_1, b_1, ..., P_3, t0_3, r_3, b_3, u1, u2, log_sigma]
    # priors per planet (loose around truth)
    planet_lo = [2.5, -0.5, 0.01, 0.0]
    planet_hi = [6.0,  2.0, 0.15, 1.0]
    LO = jnp.asarray(planet_lo * 3 + [0.0, 0.0, -10.0])
    HI = jnp.asarray(planet_hi * 3 + [1.0, 1.0,  -5.0])

    def _model(theta):
        system = keplerian.System(central=keplerian.Central(mass=1.0, radius=1.0))
        for i in range(3):
            P, t0, r, b = theta[4*i:4*(i+1)]
            system = system.add_body(period=P, time_transit=t0, radius=r,
                                      impact_param=b, eccentricity=0.0,
                                      omega_peri=0.0)
        u1, u2 = theta[12], theta[13]
        return 1.0 + limb_dark_light_curve(system, jnp.stack([u1, u2]))(t).sum(axis=1)

    def _one_lp(theta):
        m = _model(theta)
        sigma = jnp.exp(theta[14])
        resid = (flux - m) / sigma
        return -0.5 * jnp.sum(resid ** 2) - len(t) * (theta[14] + 0.5 * jnp.log(2 * jnp.pi))

    def log_prob(theta):
        in_box = jnp.all((theta >= LO) & (theta <= HI), axis=1)
        u1, u2 = theta[:, 12], theta[:, 13]
        ld_ok  = (u1 > 0) & (u1 + u2 < 1) & (u1 + 2 * u2 > 0)
        valid  = in_box & ld_ok
        lp     = jax.vmap(_one_lp)(theta)
        return jnp.where(valid, lp, -jnp.inf)

    param_names = sum(([f"P_{i+1}", f"t0_{i+1}", f"r_{i+1}", f"b_{i+1}"]
                        for i in range(3)), []) + ["u1", "u2", "log_sigma"]
    info = dict(dim=15, n_data=n_obs,
                param_names=param_names,
                prior_lo=np.asarray(LO), prior_hi=np.asarray(HI),
                truth=_TRUTH)
    return log_prob, info


def init_multi_transit(key, n_chains: int):
    """Ball near truth: 1% fractional on periods/t0/r, broader on b, u, log_sigma."""
    mean = []
    scale = []
    for i in range(3):
        mean  += [_TRUTH["P"][i], _TRUTH["t0"][i], _TRUTH["r"][i], _TRUTH["b"][i]]
        scale += [0.01, 0.01, 0.003, 0.05]
    mean  += [_TRUTH["u1"], _TRUTH["u2"], _TRUTH["log_sigma"]]
    scale += [0.05, 0.05, 0.1]
    return jnp.asarray(mean) + jnp.asarray(scale) * jax.random.normal(key, (n_chains, 15))
