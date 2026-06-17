"""Astrophysics benchmark posteriors.

Two problems, both differentiable, JAX-compatible, and widely used in the
emcee / cosmology literature:

    hogg       — Hogg, Bovy, Lang 2010 (arXiv:1008.4686) line fit with
                 outlier mixture.  D = 5, N = 20.  Toy but canonical.

    pantheon   — Pantheon+ SH0ES Type Ia SNe flat ΛCDM cosmology
                 (Scolnic+ 2022, arXiv:2202.04077).  D = 2 (Ωm, H0) with
                 analytic marginalisation over the magnitude offset M,
                 full 1701-SN covariance.

Both return a batched log-density ``(n_chains, D) -> (n_chains,)`` suitable
for any sampler in ``affine_invariant_samplers``.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Hogg, Bovy, Lang 2010 — line fit with outlier mixture
# ──────────────────────────────────────────────────────────────────────
# Table 1 of arXiv:1008.4686.  Columns: id, x, y, σy, σx, ρxy.  Points
# 1–4 are the known outliers; all 20 are fit by the mixture model.

_HOGG_TABLE1 = np.array([
    [ 1, 201, 592, 61,  9, -0.84],
    [ 2, 244, 401, 25,  4,  0.31],
    [ 3,  47, 583, 38, 11,  0.64],
    [ 4, 287, 402, 15,  7, -0.27],
    [ 5, 203, 495, 21,  5, -0.33],
    [ 6,  58, 173, 15,  9,  0.67],
    [ 7, 210, 479, 27,  4, -0.02],
    [ 8, 202, 504, 14,  4, -0.05],
    [ 9, 198, 510, 30, 11, -0.84],
    [10, 158, 416, 16,  7, -0.69],
    [11, 165, 393, 14,  5,  0.30],
    [12, 201, 442, 25,  5, -0.46],
    [13, 157, 317, 52,  5, -0.03],
    [14, 131, 311, 16,  6,  0.50],
    [15, 166, 400, 34,  6,  0.73],
    [16, 160, 337, 31,  5, -0.52],
    [17, 186, 423, 42,  9,  0.90],
    [18, 125, 334, 26,  8,  0.40],
    [19, 218, 533, 16,  6, -0.78],
    [20, 146, 344, 22,  5, -0.56],
], dtype=np.float64)


def make_hogg():
    """Hogg 2010 line fit with outlier mixture — 5-D log posterior.

    Parameters
    ----------
    θ = (m, b, log_f, μ_out, log_σ_out)

      m, b        — slope, intercept of the inlier line
      log_f       — logit of inlier fraction: f = sigmoid(log_f)
      μ_out       — mean of the Gaussian outlier component in y
      log_σ_out   — log standard deviation of the outlier component

    Likelihood is the classical Hogg mixture (eqn 17 of the paper), with
    x-errors ignored (σx is negligible compared to σy here).  Uniform
    priors on m ∈ [-5, 5], b ∈ [-200, 800], log_f ∈ [-10, 10],
    μ_out ∈ [-500, 1500], log_σ_out ∈ [0, 10].
    """
    x   = jnp.asarray(_HOGG_TABLE1[:, 1])
    y   = jnp.asarray(_HOGG_TABLE1[:, 2])
    sy  = jnp.asarray(_HOGG_TABLE1[:, 3])

    LO = jnp.asarray([-5.0, -200.0, -10.0, -500.0,  0.0])
    HI = jnp.asarray([ 5.0,  800.0,  10.0, 1500.0, 10.0])

    def log_prob(theta):
        # theta: (B, 5)
        m, b, logf, mu_o, log_s_o = (theta[:, i] for i in range(5))

        in_box = jnp.all((theta >= LO) & (theta <= HI), axis=1)

        # inlier log-likelihood per point: log N(y | m x + b, σy)
        resid_in = (y[None, :] - (m[:, None] * x[None, :] + b[:, None])) / sy[None, :]
        log_in   = -0.5 * resid_in ** 2 - jnp.log(sy[None, :]) - 0.5 * jnp.log(2 * jnp.pi)

        # outlier log-likelihood per point: log N(y | μ_out, √(σy² + σ_out²))
        s_o2    = jnp.exp(2 * log_s_o)
        var_tot = sy[None, :] ** 2 + s_o2[:, None]
        log_out = (-0.5 * (y[None, :] - mu_o[:, None]) ** 2 / var_tot
                  - 0.5 * jnp.log(2 * jnp.pi * var_tot))

        # mixture
        log_f   = -jax.nn.softplus(-logf)            # log sigmoid(logf)
        log_1mf = -jax.nn.softplus( logf)            # log (1 - sigmoid(logf))
        mix = jnp.logaddexp(log_f[:, None] + log_in, log_1mf[:, None] + log_out)

        lp = jnp.sum(mix, axis=1)
        return jnp.where(in_box, lp, -jnp.inf)

    info = dict(dim=5, n_data=len(x),
                param_names=["m", "b", "logit_f", "mu_out", "log_sigma_out"],
                prior_lo=np.asarray(LO), prior_hi=np.asarray(HI))
    return log_prob, info


def init_hogg(key, n_chains: int, mode: str = "ball"):
    """Initial walkers for the Hogg problem.

    mode = "ball"    (default) small Gaussian ball around the rough OLS
                     solution of Hogg+2010 — standard emcee practice and
                     a fair starting point for gradient-based samplers.
    mode = "prior"   uniform over the prior box; diffuse, exercises
                     burn-in.  Gradient-based HMC will *not* recover from
                     this without a stretch pre-warmup (see run_benchmark
                     ``--warm-start`` flag).
    """
    if mode == "prior":
        LO = jnp.asarray([-5.0, -200.0, -10.0, -500.0,  0.0])
        HI = jnp.asarray([ 5.0,  800.0,  10.0, 1500.0, 10.0])
        u = jax.random.uniform(key, (n_chains, 5))
        return LO + u * (HI - LO)
    # Rough OLS ≈ (2.2, 34); outlier component near (mean, log_sigma) ≈ (450, 4.5).
    mean  = jnp.asarray([2.2, 34.0, 1.4, 450.0, 4.5])
    scale = jnp.asarray([0.5, 20.0, 1.0,  50.0, 0.5])
    return mean + scale * jax.random.normal(key, (n_chains, 5))


# ──────────────────────────────────────────────────────────────────────
# Pantheon+ SH0ES — flat ΛCDM SNe Ia cosmology
# ──────────────────────────────────────────────────────────────────────
# Data: github.com/PantheonPlusSH0ES/DataRelease,
#   Pantheon+_Data/4_DISTANCES_AND_COVAR/{Pantheon+SH0ES.dat,
#                                          Pantheon+SH0ES_STAT+SYS.cov}
# Place the two files under astro/data/ .

_C_KM_S = 299_792.458  # speed of light in km/s


def _load_pantheon(data_dir: Path):
    dat_path = data_dir / "Pantheon+SH0ES.dat"
    cov_path = data_dir / "Pantheon+SH0ES_STAT+SYS.cov"
    if not dat_path.exists() or not cov_path.exists():
        raise FileNotFoundError(
            f"Pantheon+ data not found under {data_dir}. "
            "See astro/README.md — run download_pantheon.sh."
        )

    import pandas as pd
    df = pd.read_csv(dat_path, sep=r"\s+")

    # Cov file: first line is N, then N*N covariance entries row-major.
    with open(cov_path) as f:
        n = int(f.readline())
        cov = np.fromstring(f.read(), sep="\n").reshape(n, n)
    assert len(df) == n, f"cov size {n} ≠ data rows {len(df)}"
    return df, cov


def make_pantheon(data_dir: str | Path = "astro/data", z_min: float = 0.01):
    """Pantheon+ flat ΛCDM — 2-D log posterior over (Ωm, H0).

    Uses the distance modulus MU_SH0ES column and the full stat+sys
    covariance.  The per-SN absolute magnitude M is marginalised
    analytically (standard "profile likelihood" trick, e.g. Conley+ 2011):

        χ² = rᵀ C⁻¹ r − (𝟏ᵀ C⁻¹ r)² / (𝟏ᵀ C⁻¹ 𝟏)

    which is proportional to the likelihood with a flat prior on M and an
    additive constant dropped.  Cepheid-calibrated SH0ES hosts (`IS_CALIBRATOR
    == 1`) and very-low-z SNe with z < z_min are excluded — the default
    z_min = 0.01 matches standard Pantheon+ cosmology analyses.

    Parameters
    ----------
    θ = (Ωm, H0)   with priors  Ωm ∈ [0.05, 0.6],  H0 ∈ [50, 90].
    """
    data_dir = Path(data_dir)
    df, cov = _load_pantheon(data_dir)

    mask = (df["zHD"].to_numpy() > z_min) & (df["IS_CALIBRATOR"].to_numpy() == 0)
    idx  = np.where(mask)[0]
    z    = df["zHD"].to_numpy()[idx]
    mu   = df["MU_SH0ES"].to_numpy()[idx]
    C    = cov[np.ix_(idx, idx)]

    # Use float64 Cholesky for numerical stability; pass to JAX in whatever
    # precision JAX is running at (enable JAX_ENABLE_X64=1 for full f64).
    L = np.linalg.cholesky(C)
    z_j    = jnp.asarray(z)
    mu_j   = jnp.asarray(mu)
    L_j    = jnp.asarray(L)

    # Trapezoidal quadrature grid for comoving-distance integral.
    # 256 points from 0 to z_max is overkill-accurate to ~1e-6 at z~2.
    z_max = float(z.max()) * 1.01
    z_grid = jnp.linspace(0.0, z_max, 512)

    LO = jnp.asarray([0.05, 50.0])
    HI = jnp.asarray([0.60, 90.0])

    def _mu_model(Omega_m, H0):
        E = jnp.sqrt(Omega_m * (1.0 + z_grid) ** 3 + (1.0 - Omega_m))
        integrand = 1.0 / E
        cum = jnp.concatenate([
            jnp.zeros(1),
            jnp.cumsum(0.5 * (integrand[1:] + integrand[:-1])
                       * (z_grid[1:] - z_grid[:-1]))
        ])
        # D_C(z) interpolated at observed zs, in units of c/H0 → multiply.
        DC = jnp.interp(z_j, z_grid, cum) * (_C_KM_S / H0)   # Mpc
        dL = (1.0 + z_j) * DC                                 # Mpc
        return 5.0 * jnp.log10(dL) + 25.0                     # distance modulus

    def _chi2_profile(r):
        # r: residual vector, shape (N,)
        # Solve L z = r, L^T w = z  →  w = C^{-1} r, in two triangular solves.
        u = jax.scipy.linalg.solve_triangular(L_j, r, lower=True)
        # rᵀ C⁻¹ r = uᵀu
        rCr = jnp.sum(u * u)
        # 𝟏ᵀ C⁻¹ r = u0ᵀ u  where  L u0 = 𝟏
        ones = jnp.ones_like(r)
        u0 = jax.scipy.linalg.solve_triangular(L_j, ones, lower=True)
        r_one = jnp.sum(u0 * u)
        one_one = jnp.sum(u0 * u0)
        return rCr - r_one ** 2 / one_one

    def log_prob(theta):
        # theta: (B, 2)
        in_box = jnp.all((theta >= LO) & (theta <= HI), axis=1)

        def one(th):
            Om, H0 = th[0], th[1]
            mu_m = _mu_model(Om, H0)
            r    = mu_j - mu_m
            return -0.5 * _chi2_profile(r)

        lp = jax.vmap(one)(theta)
        return jnp.where(in_box, lp, -jnp.inf)

    info = dict(dim=2, n_data=len(z), z_min=z_min, z_max=float(z.max()),
                param_names=["Omega_m", "H0"],
                prior_lo=np.asarray(LO), prior_hi=np.asarray(HI))
    return log_prob, info


def init_pantheon(key, n_chains: int):
    """Initial walkers scattered near the Planck-18 point (0.315, 67.4)."""
    mean  = jnp.asarray([0.315, 67.4])
    scale = jnp.asarray([0.02,  1.5])
    return mean + scale * jax.random.normal(key, (n_chains, 2))
