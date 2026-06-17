"""3-planet transit fit in FULLY UNCONSTRAINED coordinates — the standard
exoplanet/numpyro/Stan reparameterization that removes the "hard prior
box + leapfrog divergence wall" pathology.

Same physical problem as `likelihoods_multi_transit`, same data, but
parameters are mapped to R^15:

  period   →  log period                    (strictly positive)
  t0       →  t0                            (left unconstrained; broad prior)
  r        →  log r                         (strictly positive)
  b        →  logit b                       (b ∈ [0, 1])
  (u1, u2) →  (logit q1, logit q2)          (Kipping 2013 quadratic LD
                                             reparameterization — removes
                                             the u1>0, u1+u2<1, u1+2u2>0
                                             triangle into [0,1]^2)
  log_sigma → log_sigma                     (already unbounded)

Priors become N(0, 1) × suitable scale in unconstrained space; effectively
flat on a wide range after transformation.  Jacobians are included.
HMC leapfrog now can't leap over prior boundaries — they no longer exist.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxoplanet.orbits import keplerian
from jaxoplanet.light_curves import limb_dark_light_curve


_TRUTH = dict(
    P=[3.20, 4.31, 5.19],
    t0=[0.10, 0.55, 1.20],
    r=[0.050, 0.042, 0.035],
    b=[0.40, 0.25, 0.55],
    u1=0.40, u2=0.25,
    log_sigma=np.log(200e-6),
)


def _physical_from_unconstrained(theta):
    """Map unconstrained θ ∈ R^15 → physical (P, t0, r, b, u1, u2, log_σ)."""
    # theta order (unconstrained): log_P_1, t0_1, log_r_1, logit_b_1, ..., logit_q1, logit_q2, log_sigma
    planet_params = []
    for i in range(3):
        log_P, t0, log_r, logit_b = theta[4*i:4*(i+1)]
        planet_params.append((
            jnp.exp(log_P),
            t0,
            jnp.exp(log_r),
            jax.nn.sigmoid(logit_b),
        ))
    q1 = jax.nn.sigmoid(theta[12])
    q2 = jax.nn.sigmoid(theta[13])
    # Kipping 2013 inverse transform
    sqrt_q1 = jnp.sqrt(q1)
    u1 = 2.0 * sqrt_q1 * q2
    u2 = sqrt_q1 * (1.0 - 2.0 * q2)
    log_sigma = theta[14]
    return planet_params, u1, u2, log_sigma


def _log_jac(theta):
    """Sum of log|d(physical)/d(unconstrained)| — for each transform:

        log_P   → P:     dP/dlog_P = P                → log J = log_P = theta[0]
        logit_b → b:     db/dlogit_b = b(1-b)         → log J = log b(1-b)
                                                       = -softplus(-x) - softplus(x)
        log_r   → r:     similar to log_P
        logit_q1 → q1, q1→q1:   dq1/dx = q1(1-q1)
        logit_q2 → q2:  same
        u from (q1, q2): |d(u1,u2)/d(q1,q2)| = sqrt(q1) (see Kipping)
    """
    log_jac = jnp.zeros(())
    for i in range(3):
        log_P   = theta[4*i + 0]
        logit_b = theta[4*i + 3]
        log_r   = theta[4*i + 2]
        log_jac = log_jac + log_P + log_r
        # d sigmoid(x)/dx = sigmoid(x) * (1 - sigmoid(x))
        log_jac = log_jac + (-jax.nn.softplus(-logit_b) - jax.nn.softplus(logit_b))
    # Kipping: |d(u1,u2)/d(q1,q2)| = sqrt(q1).  Combined with logit-q1/q2 → log J:
    lq1 = theta[12]; lq2 = theta[13]
    log_jac = log_jac + (-jax.nn.softplus(-lq1) - jax.nn.softplus(lq1))     # logit→q1
    log_jac = log_jac + (-jax.nn.softplus(-lq2) - jax.nn.softplus(lq2))     # logit→q2
    log_q1  = -jax.nn.softplus(-lq1)                                          # log q1
    log_jac = log_jac + 0.5 * log_q1                                          # |du/dq|=sqrt(q1)
    return log_jac


def _make_synthetic(seed: int = 0, n_obs: int = 5000, t_span: float = 30.0):
    t = np.linspace(0.0, t_span, n_obs)
    system = keplerian.System(central=keplerian.Central(mass=1.0, radius=1.0))
    for i in range(3):
        system = system.add_body(period=_TRUTH["P"][i], time_transit=_TRUTH["t0"][i],
                                  radius=_TRUTH["r"][i], impact_param=_TRUTH["b"][i],
                                  eccentricity=0.0, omega_peri=0.0)
    lc = np.asarray(limb_dark_light_curve(system, [_TRUTH["u1"], _TRUTH["u2"]])(t))
    model = 1.0 + lc.sum(axis=1)
    sigma = np.exp(_TRUTH["log_sigma"])
    rng = np.random.default_rng(seed)
    flux = model + sigma * rng.standard_normal(n_obs)
    return t, flux


def make_multi_transit_reparam(seed: int = 0, n_obs: int = 5000):
    t_np, flux_np = _make_synthetic(seed=seed, n_obs=n_obs)
    t    = jnp.asarray(t_np)
    flux = jnp.asarray(flux_np)

    # Broad Gaussian priors in unconstrained space — ~flat over realistic range.
    prior_sigma = jnp.asarray([1.0, 0.5, 1.0, 3.0,
                                1.0, 0.5, 1.0, 3.0,
                                1.0, 0.5, 1.0, 3.0,
                                3.0, 3.0, 2.0])
    prior_mean  = jnp.asarray([np.log(_TRUTH["P"][0]), _TRUTH["t0"][0],
                                np.log(_TRUTH["r"][0]),
                                np.log(_TRUTH["b"][0] / (1 - _TRUTH["b"][0])),
                                np.log(_TRUTH["P"][1]), _TRUTH["t0"][1],
                                np.log(_TRUTH["r"][1]),
                                np.log(_TRUTH["b"][1] / (1 - _TRUTH["b"][1])),
                                np.log(_TRUTH["P"][2]), _TRUTH["t0"][2],
                                np.log(_TRUTH["r"][2]),
                                np.log(_TRUTH["b"][2] / (1 - _TRUTH["b"][2])),
                                0.0, 0.0,
                                _TRUTH["log_sigma"]])

    def _one_lp(theta):
        planet_params, u1, u2, log_sigma = _physical_from_unconstrained(theta)
        system = keplerian.System(central=keplerian.Central(mass=1.0, radius=1.0))
        for P, t0, r, b in planet_params:
            system = system.add_body(period=P, time_transit=t0, radius=r,
                                      impact_param=b, eccentricity=0.0,
                                      omega_peri=0.0)
        m = 1.0 + limb_dark_light_curve(system, jnp.stack([u1, u2]))(t).sum(axis=1)
        sigma = jnp.exp(log_sigma)
        resid = (flux - m) / sigma
        log_lik = -0.5 * jnp.sum(resid ** 2) - len(t) * (log_sigma + 0.5 * jnp.log(2 * jnp.pi))
        log_prior = -0.5 * jnp.sum(((theta - prior_mean) / prior_sigma) ** 2)
        return log_lik + log_prior + _log_jac(theta)

    def log_prob(theta):
        return jax.vmap(_one_lp)(theta)

    param_names = sum(([f"log_P_{i+1}", f"t0_{i+1}",
                         f"log_r_{i+1}", f"logit_b_{i+1}"] for i in range(3)), []) + \
                  ["logit_q1", "logit_q2", "log_sigma"]
    info = dict(dim=15, n_data=n_obs,
                param_names=param_names,
                prior_lo=np.asarray(prior_mean - 10 * prior_sigma),
                prior_hi=np.asarray(prior_mean + 10 * prior_sigma),
                truth=_TRUTH)
    return log_prob, info


def init_multi_transit_reparam(key, n_chains: int):
    """Ball near truth in unconstrained coords."""
    mean = []
    for i in range(3):
        mean += [np.log(_TRUTH["P"][i]), _TRUTH["t0"][i],
                  np.log(_TRUTH["r"][i]),
                  np.log(_TRUTH["b"][i] / (1 - _TRUTH["b"][i]))]
    mean += [0.0, 0.0, _TRUTH["log_sigma"]]          # q1, q2 start at 0.5
    mean = jnp.asarray(mean)
    scale = jnp.asarray([0.01, 0.01, 0.05, 0.1,
                          0.01, 0.01, 0.05, 0.1,
                          0.01, 0.01, 0.05, 0.1,
                          0.1, 0.1, 0.1])
    return mean + scale * jax.random.normal(key, (n_chains, 15))
