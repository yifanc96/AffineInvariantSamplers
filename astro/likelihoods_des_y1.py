"""DES Y1 3×2pt mock likelihood — 21-D cosmology + nuisance, fully JAX.

Port of Campagne+ 2023 (JAX-COSMO, arXiv:2302.05163) DES Y1 setup.  Uses
the released ``2pt_NG_mcal_1110.fits`` to load the source / lens n(z)
histograms, then treats the theory mean at fiducial parameters as the
"data" (standard mock-data setup in forecasting).  Gaussian likelihood
against a fixed Fisher-derived covariance.

**Parameters (21-D, unconstrained space)** after non-centered reparam
matching the jax-cosmo-paper ``VectorizedNumPyro_block`` notebook:

    Ω_c, σ_8, Ω_b, h, n_s, w_0                 (cosmology, uniform priors)
    A_IA, η_IA                                  (intrinsic alignment)
    b_1, b_2, b_3, b_4, b_5                     (linear galaxy bias per lens bin)
    m_1, m_2, m_3, m_4                          (shear multiplicative calibration)
    Δz_1, Δz_2, Δz_3, Δz_4                      (source photo-z shifts)

Uniform-prior parameters use a **logit re-parameterisation** θ = lo +
(hi-lo)·σ(z); Normal-prior parameters use **non-centered** θ = μ + σ·z.
In z-space the prior is product of N(0,1) for Normal dims and a logit
Jacobian for Uniform dims.  HMC sees a smooth, near-Gaussian target.

The "affine-invariant" story: peaches / peanuts use the 21×21 ensemble
covariance as full preconditioner — captures A_IA↔Δz, cosmology↔bias,
S_8 banana cross-correlations that Campagne+'s *block-dense* NUTS
misses (their blocks: {cosmology 6D}, {bias 5D}, singletons elsewhere).
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np


# Uniform priors: (lo, hi)
_UNIF = [
    ("Omega_c", 0.1, 0.9),
    ("sigma8",  0.4, 1.0),
    ("Omega_b", 0.03, 0.07),
    ("h",       0.55, 0.91),
    ("n_s",     0.87, 1.07),
    ("w0",      -2.0, -0.33),
    ("A",       -5.0, 5.0),
    ("eta",     -5.0, 5.0),
    ("b1",      0.8, 3.0),
    ("b2",      0.8, 3.0),
    ("b3",      0.8, 3.0),
    ("b4",      0.8, 3.0),
    ("b5",      0.8, 3.0),
]
# Normal priors: (μ, σ)  — from DES Y1 systematics paper
_NORM = [
    ("m1",  0.012,  0.023),
    ("m2",  0.012,  0.023),
    ("m3",  0.012,  0.023),
    ("m4",  0.012,  0.023),
    ("dz1",  0.001,  0.016),
    ("dz2", -0.019,  0.013),
    ("dz3",  0.009,  0.011),
    ("dz4", -0.018,  0.022),
]
_DIM = len(_UNIF) + len(_NORM)                          # = 21

_UNIF_LO = jnp.asarray([lo for _, lo, _ in _UNIF])
_UNIF_HI = jnp.asarray([hi for _, _, hi in _UNIF])
_NORM_MU = jnp.asarray([mu for _, mu, _ in _NORM])
_NORM_SD = jnp.asarray([sd for _, _, sd in _NORM])

# Fiducial cosmology for generating the mock data
_FID = dict(
    Omega_c=0.2545, sigma8=0.801, Omega_b=0.0485,
    h=0.682, n_s=0.971, w0=-1.0,
    A=0.5, eta=0.0,
    b=[1.2, 1.4, 1.6, 1.8, 2.0],
    m=[0., 0., 0., 0.],
    dz=[0., 0., 0., 0.],
)


def _z_to_physical(z):
    """Unconstrained z ∈ R^21 → physical parameter vector (21-D)."""
    n_u = len(_UNIF)
    phys_u = _UNIF_LO + (_UNIF_HI - _UNIF_LO) * jax.nn.sigmoid(z[:n_u])
    phys_n = _NORM_MU + _NORM_SD * z[n_u:]
    return jnp.concatenate([phys_u, phys_n])


def _log_jacobian(z):
    """Sum of log|dθ_i/dz_i|.  Uniform (logit) gives
       (hi-lo)·σ(z)·(1-σ(z));  Normal (non-centered) gives σ_i  (constant)."""
    n_u = len(_UNIF)
    zu = z[:n_u]
    log_u = (jnp.log(_UNIF_HI - _UNIF_LO)
              - jax.nn.softplus(-zu) - jax.nn.softplus(zu))
    log_n = jnp.log(_NORM_SD)                          # constant in z
    return jnp.sum(log_u) + jnp.sum(log_n)


def _theory_and_data():
    """Build the jax-cosmo forward model + fiducial mock data vector."""
    import jax_cosmo as jc
    from astropy.io import fits

    data_path = os.path.join(os.path.dirname(__file__), "data",
                              "2pt_NG_mcal_1110.fits")
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"DES Y1 data file missing. Run:  cd astro/data/ && wget "
            f"http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/"
            f"chains/2pt_NG_mcal_1110.fits")
    with fits.open(data_path) as hdul:
        nz_source = hdul[6].data
        nz_lens   = hdul[7].data

    neff_s = [1.47, 1.46, 1.50, 0.73]
    nzs_s = [jc.redshift.kde_nz(nz_source["Z_MID"].astype("float32"),
                                 nz_source[f"BIN{i}"].astype("float32"),
                                 bw=0.01, gals_per_arcmin2=neff_s[i-1])
             for i in range(1, 5)]
    nzs_l = [jc.redshift.kde_nz(nz_lens["Z_MID"].astype("float32"),
                                 nz_lens[f"BIN{i}"].astype("float32"),
                                 bw=0.01)
             for i in range(1, 6)]
    ell = jnp.logspace(1, 3, 50)

    def probes_from(cosmo, m, dz, A, eta, bias):
        nzs_s_sys = [jc.redshift.systematic_shift(nzi, dzi, zmax=2.0)
                     for nzi, dzi in zip(nzs_s, dz)]
        b_ia = jc.bias.des_y1_ia_bias(A, eta, 0.62)
        b_g  = [jc.bias.constant_linear_bias(bi) for bi in bias]
        return [jc.probes.WeakLensing(nzs_s_sys,
                                       ia_bias=b_ia,
                                       multiplicative_bias=m),
                jc.probes.NumberCounts(nzs_l, b_g)]

    def theory_mean_from_phys(phys):
        cosmo = jc.Cosmology(Omega_c=phys[0], sigma8=phys[1], Omega_b=phys[2],
                              h=phys[3], n_s=phys[4], w0=phys[5],
                              Omega_k=0.0, wa=0.0)
        A, eta = phys[6], phys[7]
        bias = [phys[8], phys[9], phys[10], phys[11], phys[12]]
        m    = [phys[13], phys[14], phys[15], phys[16]]
        dz   = [phys[17], phys[18], phys[19], phys[20]]
        probes = probes_from(cosmo, m, dz, A, eta, bias)
        cl = jc.angular_cl.angular_cl(cosmo, ell, probes)
        return cl.flatten()

    # Build fiducial phys vector
    fid_phys = jnp.asarray(
        [_FID["Omega_c"], _FID["sigma8"], _FID["Omega_b"],
         _FID["h"], _FID["n_s"], _FID["w0"],
         _FID["A"], _FID["eta"],
         *_FID["b"], *_FID["m"], *_FID["dz"]])

    def theory_cov_from_phys(phys):
        cosmo = jc.Cosmology(Omega_c=phys[0], sigma8=phys[1], Omega_b=phys[2],
                              h=phys[3], n_s=phys[4], w0=phys[5],
                              Omega_k=0.0, wa=0.0)
        A, eta = phys[6], phys[7]
        bias = [phys[8], phys[9], phys[10], phys[11], phys[12]]
        m    = [phys[13], phys[14], phys[15], phys[16]]
        dz   = [phys[17], phys[18], phys[19], phys[20]]
        probes = probes_from(cosmo, m, dz, A, eta, bias)
        cl_signal = jc.angular_cl.angular_cl(cosmo, ell, probes)
        cl_noise  = jc.angular_cl.noise_cl(ell, probes)
        cov = jc.angular_cl.gaussian_cl_covariance(
            ell, probes, cl_signal, cl_noise, f_sky=0.25, sparse=False)
        return cov

    print("[des_y1] Computing mock data + covariance at fiducial Λ…")
    data_mean = theory_mean_from_phys(fid_phys)
    cov = theory_cov_from_phys(fid_phys)
    return theory_mean_from_phys, data_mean, cov, fid_phys


def make_des_y1():
    theory_fn, data_mean, cov, fid_phys = _theory_and_data()

    # Pre-compute Cholesky for a stable Gaussian likelihood.
    cov_np = np.asarray(cov, dtype=np.float64)
    # Symmetrise + regularise.
    cov_np = 0.5 * (cov_np + cov_np.T) + 1e-10 * np.eye(cov_np.shape[0])
    L = np.linalg.cholesky(cov_np)
    L_j = jnp.asarray(L)
    data_mean_j = jnp.asarray(data_mean)

    def _one_lp(z):
        phys = _z_to_physical(z)
        cl = theory_fn(phys)
        r  = data_mean_j - cl
        u  = jax.scipy.linalg.solve_triangular(L_j, r, lower=True)
        log_lik = -0.5 * jnp.sum(u * u)
        # Prior in z-space: N(0,1) for Normal-reparameterised dims;
        # the Uniform-prior dims have a constant log-prior inside their box,
        # and we include their Jacobian so that the posterior in z-space
        # matches the physical posterior.
        n_u = len(_UNIF)
        log_prior_z = -0.5 * jnp.sum(z[n_u:] ** 2)   # N(0,1) on Normal dims
        return log_lik + log_prior_z + _log_jacobian(z)

    def log_prob(theta):
        # Accept (D,) for single-walker gradients (malt), (B, D) for batched.
        if theta.ndim == 1:
            return _one_lp(theta)
        return jax.vmap(_one_lp)(theta)

    # Residual for Gauss-Newton preconditioning (gndr).  log p(z) ≈
    #   −0.5 ‖r(z)‖²  up to the Jacobian + uniform-prior contributions,
    # which are neglected (the 2250-element likelihood dominates in the GN
    # metric).  Including Normal-prior z-dims as extra residual entries so
    # GN also captures their unit curvature.
    n_u = len(_UNIF)
    def residual_fn(z):
        """(D,) → (n_data + n_norm,).  log p ≈ −½‖r‖² (modulo Jacobian)."""
        phys = _z_to_physical(z)
        cl = theory_fn(phys)
        r_lik  = jax.scipy.linalg.solve_triangular(L_j, data_mean_j - cl, lower=True)
        r_norm = z[n_u:]                                 # N(0,1) prior residuals
        return jnp.concatenate([r_lik, r_norm])

    # Fiducial z vector (inverse-transform of fid_phys) for init + priors info
    n_u = len(_UNIF)
    z_fid_u = jnp.asarray(
        jnp.log((fid_phys[:n_u] - _UNIF_LO) / (_UNIF_HI - fid_phys[:n_u])))
    z_fid_n = (fid_phys[n_u:] - _NORM_MU) / _NORM_SD
    z_fid = jnp.concatenate([z_fid_u, z_fid_n])

    param_names = [n for n, _, _ in _UNIF] + [n for n, _, _ in _NORM]
    # Prior lo/hi in z-space: soft 10-sigma bounds for book-keeping
    prior_lo = np.full(_DIM, -10.0)
    prior_hi = np.full(_DIM,  10.0)
    info = dict(dim=_DIM, n_data=int(data_mean.size),
                param_names=param_names,
                prior_lo=prior_lo, prior_hi=prior_hi,
                z_fid=np.asarray(z_fid),
                residual_fn=residual_fn)
    return log_prob, info


def init_des_y1(key, n_chains: int, init_scale: float = 0.1,
                 init_mode: str = "fid_ball"):
    """Initial walker positions in z-space.

    Parameters
    ----------
    init_scale : float, default 0.1
        Std of the Gaussian ball.  ``0.1`` is a "tight ball" requiring stretch
        pre-warmup; ``0.5–1.0`` is "wide" and already covers most of the
        typical set.
    init_mode : {"fid_ball", "prior_median"}, default "fid_ball"
        Center of the ball.  ``"fid_ball"`` centers at the fiducial truth in
        z-space (an oracle warm-start; what our earlier tables used).
        ``"prior_median"`` centers at z = 0 (the prior median in unconstrained
        space — Uniform(lo,hi) median maps to logit(0.5) = 0; Normal mean →
        0).  ``"prior_median"`` matches numpyro's NUTS ``init_to_median``
        center exactly and is the fair-comparison default for synthetic-data
        benchmarks.
    """
    if init_mode == "fid_ball":
        fid_phys_u = (jnp.asarray([_FID["Omega_c"], _FID["sigma8"], _FID["Omega_b"],
                                    _FID["h"],       _FID["n_s"],   _FID["w0"],
                                    _FID["A"],       _FID["eta"],
                                    *_FID["b"]]))
        z_center_u = jnp.log((fid_phys_u - _UNIF_LO) / (_UNIF_HI - fid_phys_u))
        z_center_n = jnp.zeros(len(_NORM))           # truth m/dz ≈ 0 so z≈0
        z_center = jnp.concatenate([z_center_u, z_center_n])
    elif init_mode == "prior_median":
        z_center = jnp.zeros(_DIM)                   # logit(0.5)=0; Normal mean=0
    else:
        raise ValueError(f"unknown init_mode={init_mode!r}")
    return z_center[None, :] + init_scale * jax.random.normal(key, (n_chains, _DIM))
