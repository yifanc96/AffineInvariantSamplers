"""Gauss-Newton residual for alanine dipeptide.

The residual is built from the *harmonic-expressible* components of the
AMBER + GBSA potential:

    r_bond   = √(k_b/kT) · (d_ij − d_b⁰)                   one per bond
    r_angle  = √(k_a/kT) · (θ_ijk − θ_a⁰)                  one per angle
    r_iso    = √(1/σ²_iso · 1/kT) · (x_iα − x_ref_iα)      isotropic prior

Torsion / LJ / Coulomb / GBSA are *not* folded into r(x) — they enter via
the full log-prob (OpenMM callback) and the MH acceptance corrects for
the residual mismatch.  This follows the same "harmonic-only preconditioner,
full log-prob target" recipe we used for LJ-13.
"""
from __future__ import annotations

import numpy as np
import jax, jax.numpy as jnp
import openmm


def extract_bonded_params(system: openmm.System):
    """Pull (atom indices, force constant, equilibrium value) tuples from
    HarmonicBondForce and HarmonicAngleForce.  Returns pure np arrays."""
    bonds = []; angles = []
    for f in system.getForces():
        name = type(f).__name__
        if name == "HarmonicBondForce":
            for i in range(f.getNumBonds()):
                a, b, d0, k = f.getBondParameters(i)
                bonds.append((a, b,
                               d0.value_in_unit_system(openmm.unit.md_unit_system),
                               k .value_in_unit_system(openmm.unit.md_unit_system)))
        elif name == "HarmonicAngleForce":
            for i in range(f.getNumAngles()):
                a, b, c, t0, k = f.getAngleParameters(i)
                angles.append((a, b, c,
                                 t0.value_in_unit_system(openmm.unit.md_unit_system),
                                 k .value_in_unit_system(openmm.unit.md_unit_system)))
    bonds  = np.asarray(bonds, dtype=np.float64)        # (Nb, 4) [a, b, d0, k]
    angles = np.asarray(angles, dtype=np.float64)       # (Na, 5)
    return bonds, angles


def make_alanine_residual(system: openmm.System, x_ref_flat,
                            kT: float, sigma_iso: float = 0.3):
    """Return a JAX residual_fn(x_flat) -> (N_res,)  for (D,) input.

    Args:
        system    : OpenMM system (used for parameter extraction only).
        x_ref_flat: (D,) reference coordinates (nm).  r_iso anchors here.
        kT        : thermal energy in kJ/mol (for normalisation).
        sigma_iso : isotropic prior length scale (nm).  Large ⇒ weak.
    """
    bonds, angles = extract_bonded_params(system)
    n_atoms = system.getNumParticles()

    bond_i  = jnp.asarray(bonds[:, 0].astype(np.int32))
    bond_j  = jnp.asarray(bonds[:, 1].astype(np.int32))
    bond_d0 = jnp.asarray(bonds[:, 2])
    bond_c  = jnp.sqrt(jnp.asarray(bonds[:, 3]) / kT)

    ang_i   = jnp.asarray(angles[:, 0].astype(np.int32))
    ang_j   = jnp.asarray(angles[:, 1].astype(np.int32))
    ang_k   = jnp.asarray(angles[:, 2].astype(np.int32))
    ang_t0  = jnp.asarray(angles[:, 3])
    ang_c   = jnp.sqrt(jnp.asarray(angles[:, 4]) / kT)

    x_ref = jnp.asarray(x_ref_flat)
    iso_c = jnp.sqrt(1.0 / (sigma_iso ** 2 * kT))

    def residual_fn(x_flat):
        x = x_flat.reshape(n_atoms, 3)

        d_ij = jnp.linalg.norm(x[bond_i] - x[bond_j], axis=-1)
        r_b  = bond_c * (d_ij - bond_d0)

        b1 = x[ang_i] - x[ang_j]
        b2 = x[ang_k] - x[ang_j]
        b1n = b1 / (jnp.linalg.norm(b1, axis=-1, keepdims=True) + 1e-12)
        b2n = b2 / (jnp.linalg.norm(b2, axis=-1, keepdims=True) + 1e-12)
        cos_t = jnp.clip((b1n * b2n).sum(axis=-1), -1.0 + 1e-9, 1.0 - 1e-9)
        r_a = ang_c * (jnp.arccos(cos_t) - ang_t0)

        r_iso = iso_c * (x_flat - x_ref)

        return jnp.concatenate([r_b, r_a, r_iso])

    info = dict(n_bonds=int(bonds.shape[0]), n_angles=int(angles.shape[0]),
                 n_iso=3 * n_atoms)
    return residual_fn, info
