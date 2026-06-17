"""MD benchmark energy functions: DW-4, LJ-13, LJ-55.

DW-4 (4 particles in 2D, d = 8)

    E(x) = Σ_{i<j} [ a (d_ij - d_0) + b (d_ij - d_0)² + c (d_ij - d_0)⁴ ]
    a=0, b=-4, c=0.9, d_0=4, τ=1

LJ-n (n particles in 3D, d = 3n)  — n=13 or n=55

Two formulas exist in the literature and DIFFER by a factor of 2 on the LJ term:

  iDEM / Adjoint-Sampling / bgflow class — formula as written in the paper:

      E_tot(x) = E_LJ(x) + E_osc(x)
      E_LJ(x)  = Σ_{i<j} [ (r_m/d_ij)^12 − 2 (r_m/d_ij)^6 ]
      E_osc(x) = (1/2) Σ_i ‖x_i − x̄‖²
      log π(x) = − E_tot(x) / τ     with τ=1, r_m=ε=1

  pita MCMC script (github.com/taraak/pita, arXiv:2506.16471) — formula that
  actually generated the published DEM / iDEM / Adjoint-Sampling reference data:

      E_tot(x) = 2·E_LJ(x) + E_osc(x)
      log π(x) = − E_tot(x) / τ     with τ=1

Under the pita target the MCMC reference (DEM `.npy` files) is reproduced exactly
with no τ calibration.  See ``MD/README.md``.  Samplers in this repo default to
the iDEM formula; pass ``formula="pita"`` (or ``--formula pita`` on the CLI) to
use the correct-GT version.

Samplers expect a batched log-density of shape (n_chains, D) → (n_chains,).
The returned log_prob is -E.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def _pairwise_distances(coords: jax.Array) -> jax.Array:
    """Upper-triangular pairwise distances.

    coords: (..., n, dim) → (..., n*(n-1)/2)
    Numerically stabilised sqrt avoids NaN gradients at d=0.
    """
    diffs = coords[..., :, None, :] - coords[..., None, :, :]      # (..., n, n, dim)
    sq    = jnp.sum(diffs * diffs, axis=-1)                        # (..., n, n)
    n     = coords.shape[-2]
    iu, ju = jnp.triu_indices(n, k=1)
    sq_upper = sq[..., iu, ju]                                     # (..., n*(n-1)/2)
    return jnp.sqrt(sq_upper + 1e-12)


# ──────────────────────────────────────────────────────────────────────
# DW-4
# ──────────────────────────────────────────────────────────────────────

def make_dw4(a: float = 0.0, b: float = -4.0, c: float = 0.9,
             d0: float = 4.0, tau: float = 1.0):
    """Double-well over 4 particles in 2D  (D = 8).

    Returns log_prob_batched : (B, 8) → (B,).
    """
    n_part, dim = 4, 2

    def log_prob(x: jax.Array) -> jax.Array:
        coords = x.reshape(x.shape[0], n_part, dim)
        d = _pairwise_distances(coords)                          # (B, 6)
        dd = d - d0
        E = (a * dd + b * dd**2 + c * dd**4).sum(axis=-1) / tau  # (B,)
        return -E

    return log_prob


# ──────────────────────────────────────────────────────────────────────
# Lennard-Jones (LJ-13, LJ-55)
# ──────────────────────────────────────────────────────────────────────

def make_lj(n_particles: int, rm: float = 1.0, eps: float = 1.0,
            tau: float = 1.0, osc_scale: float = 1.0):
    """iDEM / Adjoint-Sampling formula as written in the papers:
    E = E_LJ + osc_scale · E_osc  (no factor of 2 on LJ).

    Does NOT reproduce the DEM reference dataset — see ``make_lj_pita``
    for the target that actually generated those `.npy` files."""
    n_part, dim = n_particles, 3

    def log_prob(x: jax.Array) -> jax.Array:
        coords = x.reshape(x.shape[0], n_part, dim)
        d = _pairwise_distances(coords)
        inv6  = (rm / d) ** 6
        inv12 = inv6 * inv6
        E_lj  = eps * (inv12 - 2.0 * inv6).sum(axis=-1)

        com    = coords.mean(axis=-2, keepdims=True)
        E_osc  = 0.5 * jnp.sum((coords - com) ** 2, axis=(-2, -1))

        return -(E_lj + osc_scale * E_osc) / tau

    return log_prob


# ──────────────────────────────────────────────────────────────────────
# pita (arXiv:2506.16471) convention — the actual GT-generating formula.
# From `sampling/sample_lj13.py`:  E = 2·E_LJ + E_osc, τ=1.
# ──────────────────────────────────────────────────────────────────────

def make_lj_pita(n_particles: int, rm: float = 1.0, eps: float = 1.0,
                  tau: float = 1.0, osc_scale: float = 1.0,
                  smooth: bool = False, r_cut: float = 0.65):
    """E(x) = 2·E_LJ(x) + osc_scale·E_osc(x),  log π = −E / τ.

    ``smooth=True`` uses the soft-core LJ below ``r_cut``."""
    n_part, dim = n_particles, 3

    def log_prob(x: jax.Array) -> jax.Array:
        coords = x.reshape(x.shape[0], n_part, dim)
        d = _pairwise_distances(coords)
        if smooth:
            lj_pair = _lj_pair_smooth(d, rm=rm, eps=eps, r_cut=r_cut)
        else:
            inv6  = (rm / d) ** 6
            inv12 = inv6 * inv6
            lj_pair = eps * (inv12 - 2.0 * inv6)
        E_lj  = lj_pair.sum(axis=-1)

        com    = coords.mean(axis=-2, keepdims=True)
        E_osc  = 0.5 * jnp.sum((coords - com) ** 2, axis=(-2, -1))

        return -(2.0 * E_lj + osc_scale * E_osc) / tau

    return log_prob


def _lj_pair_smooth(d, rm=1.0, eps=1.0, r_cut=0.65):
    """LJ pair energy with soft-core below ``r_cut``.

    Below `r_cut` the raw `ε·[(r_m/r)^12 − 2(r_m/r)^6]` is replaced by a
    quadratic extrapolation that matches value and derivative at `r_cut`
    — equivalent in spirit to pita's ``LennardJonesPotential(smooth=True)``.
    Leaves the populated region (r ≳ 0.85 · r_m) untouched; regularises
    only the singular wall so MH samplers can't get stuck there.
    """
    inv6  = (rm / d) ** 6
    inv12 = inv6 * inv6
    raw   = eps * (inv12 - 2.0 * inv6)

    inv6_c  = (rm / r_cut) ** 6
    inv12_c = inv6_c * inv6_c
    V_c     = eps * (inv12_c - 2.0 * inv6_c)
    dV_c    = eps * (-12.0 * inv12_c / r_cut + 12.0 * inv6_c / r_cut)
    # Quadratic continuation: V(r) = V_c + dV_c·(r−r_cut) + ½·k·(r−r_cut)²,
    # k chosen so slope at r=0 is finite (gentle wall).
    k_soft  = 6.0 * eps / (rm * rm)
    soft    = V_c + dV_c * (d - r_cut) + 0.5 * k_soft * (d - r_cut) ** 2
    return jnp.where(d < r_cut, soft, raw)


def lj_pita_energy(x, n_particles, rm=1.0, eps=1.0, tau=1.0, osc_scale=1.0,
                    smooth=False, r_cut=0.65):
    """E(x) = 2·E_LJ(x) + osc_scale·E_osc(x)  (evaluated at τ=1 by default).

    ``smooth=True`` replaces the singular `1/r^12` wall below `r_cut` with
    a quadratic continuation (pita's ``LennardJonesPotential(smooth=True)``
    style; see ``_lj_pair_smooth``).  Samples in the populated region are
    unchanged; MH samplers stop getting stuck in the `r → 0` pathology."""
    coords = x.reshape(x.shape[0], n_particles, 3)
    d = _pairwise_distances(coords)
    if smooth:
        lj_pair = _lj_pair_smooth(d, rm=rm, eps=eps, r_cut=r_cut)
    else:
        inv6  = (rm / d) ** 6
        inv12 = inv6 * inv6
        lj_pair = eps * (inv12 - 2.0 * inv6)
    E_lj  = lj_pair.sum(-1)
    com    = coords.mean(axis=-2, keepdims=True)
    E_osc  = 0.5 * jnp.sum((coords - com) ** 2, axis=(-2, -1))
    return (2.0 * E_lj + osc_scale * E_osc) / tau


def project_com_zero(x: jax.Array, n_particles: int, spatial_dim: int = 3) -> jax.Array:
    """Subtract the per-particle mean so the COM of each sample is at origin.

    ``x`` has shape (B, n_particles * spatial_dim); returns same shape.
    """
    coords = x.reshape(x.shape[0], n_particles, spatial_dim)
    coords = coords - coords.mean(axis=-2, keepdims=True)
    return coords.reshape(x.shape[0], n_particles * spatial_dim)


# ──────────────────────────────────────────────────────────────────────
# Per-sample energy (useful for histograms / metrics)
# ──────────────────────────────────────────────────────────────────────

def dw4_energy(x, a=0.0, b=-4.0, c=0.9, d0=4.0, tau=1.0):
    coords = x.reshape(x.shape[0], 4, 2)
    d = _pairwise_distances(coords)
    dd = d - d0
    return (a * dd + b * dd**2 + c * dd**4).sum(-1) / tau


def lj_energy(x, n_particles, rm=1.0, eps=1.0, tau=1.0, osc_scale=1.0):
    """Evaluator matching ``make_lj``:  E = (E_LJ + osc_scale·E_osc) / τ."""
    coords = x.reshape(x.shape[0], n_particles, 3)
    d = _pairwise_distances(coords)
    inv6  = (rm / d) ** 6
    inv12 = inv6 * inv6
    E_lj  = eps * (inv12 - 2.0 * inv6).sum(-1)
    com = coords.mean(-2, keepdims=True)
    E_osc = 0.5 * jnp.sum((coords - com) ** 2, axis=(-2, -1))
    return (E_lj + osc_scale * E_osc) / tau


# ──────────────────────────────────────────────────────────────────────
# Initial configurations
# ──────────────────────────────────────────────────────────────────────

def init_dw4(key, n_chains: int, scale: float = 2.0) -> jax.Array:
    """Random Gaussian init centred at origin; DW-4 minima lie near d_ij ≈ 4 ± 1.5.

    With scale=2.0 the average pairwise distance ≈ 2·scale·√2 ≈ 5.6, already
    near the outer well (d₀ + 1.49 = 5.49), giving finite initial energy.
    """
    return scale * jax.random.normal(key, (n_chains, 8))


def init_lj(key, n_chains: int, n_particles: int,
            spacing: float = 1.3, perturb: float = 0.2) -> jax.Array:
    """Random ‘shaken’ cubic lattice with COM at origin.

    Nearest-neighbour distance is ``spacing`` (> r_m=1 to keep the 1/r^12
    repulsion finite) and Gaussian noise ``perturb`` breaks the symmetry
    across walkers.  ``perturb`` should be large enough to give the ensemble
    non-trivial covariance (otherwise B = centered/√N shrinks to zero and
    the affine-invariant preconditioner stalls the sampler).
    """
    import numpy as np
    side = int(np.ceil(n_particles ** (1 / 3)))
    xs, ys, zs = np.meshgrid(np.arange(side), np.arange(side), np.arange(side),
                             indexing="ij")
    grid = spacing * np.stack([xs.ravel(), ys.ravel(), zs.ravel()],
                              axis=1).astype(np.float32)
    grid = grid[:n_particles]
    grid -= grid.mean(axis=0, keepdims=True)                       # COM at 0

    coords_base = jnp.asarray(grid)[None]                          # (1, n, 3)
    coords_base = jnp.broadcast_to(coords_base, (n_chains, n_particles, 3))
    noise = perturb * jax.random.normal(key, coords_base.shape)
    coords = coords_base + noise
    return coords.reshape(n_chains, n_particles * 3)
