"""Alanine dipeptide setup (ASBS / Midgley 2023 benchmark).

Builds the OpenMM system from ``bgmol.systems.ala2.AlanineDipeptideImplicit``
(Amber ff + GB-OBC implicit solvent), exposes:

    - ``build_system()``       — returns (topology, system, positions_nm)
    - ``openmm_energy(...)``   — NumPy batched energy / force wrapper
    - ``jax_log_prob(...)``    — JAX pure_callback wrapper  (batched → scalar log-prob)
    - ``langevin_sample(...)`` — native OpenMM Langevin integrator (reference-style)

Temperature convention: all our samplers use log_prob = −E/τ with τ=1.
For alanine at T=300 K, we divide the OpenMM energy (kJ/mol) by
kT = R·T = 8.314 × 300 / 1000 ≈ 2.479 kJ/mol.  After that τ=1 is correct.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

import openmm
import openmm.unit as unit

# bgmol imports (with a few local patches applied at import time in the MD/
# folder — see MD/README.md).
from bgmol.systems.ala2 import AlanineDipeptideImplicit


kT_kJmol_300K = 8.314462618e-3 * 300.0           # kJ/mol at 300 K ≈ 2.494


# ──────────────────────────────────────────────────────────────────────
# System construction
# ──────────────────────────────────────────────────────────────────────

def build_system():
    """Returns (topology, openmm.System, positions_np_nm, atom_names)."""
    s = AlanineDipeptideImplicit()
    positions = np.asarray([[v.x, v.y, v.z]
                             for v in s.positions._value]) \
                if hasattr(s.positions, "_value") else np.asarray(s.positions)
    positions = positions.astype(np.float64)
    atoms = [atom.name for atom in s.topology.atoms()]
    return s.topology, s.system, positions, atoms


def phi_psi_atom_indices(topology):
    """Return two tuples of atom indices (4-tuples) for (phi, psi) torsions.

    φ: C(ACE) – N(ALA) – CA – C(ALA)
    ψ: N(ALA) – CA – C(ALA) – N(NME)

    Resolved by residue id because atom names repeat across caps and ALA.
    """
    by_res = {}
    for a in topology.atoms():
        by_res.setdefault(a.residue.index, {})[a.name] = a.index
    C_ace = by_res[0]["C"]
    N_ala = by_res[1]["N"]; CA = by_res[1]["CA"]; C_ala = by_res[1]["C"]
    N_nme = by_res[2]["N"]
    phi = (C_ace, N_ala, CA, C_ala)
    psi = (N_ala, CA, C_ala, N_nme)
    return phi, psi


# ──────────────────────────────────────────────────────────────────────
# OpenMM energy / force evaluator (batched, NumPy)
# ──────────────────────────────────────────────────────────────────────

class OpenMMEnergy:
    """Per-chain OpenMM Context pool; batched energy+force evaluator.

    Each chain gets its own Context so we can evaluate a batch without
    re-setting positions on a single shared context.
    """
    def __init__(self, system: openmm.System, n_chains: int = 1,
                 platform_name: str = "OpenCL"):
        self.system = system
        self.n_atoms = system.getNumParticles()
        self.n_chains = n_chains
        self.platform = openmm.Platform.getPlatformByName(platform_name)
        # Dummy integrator — we'll compute energies via Context directly
        self.contexts = []
        for _ in range(n_chains):
            integ = openmm.VerletIntegrator(1e-3)
            self.contexts.append(openmm.Context(system, integ, self.platform))

    def energy_forces(self, coords_batch_nm: np.ndarray):
        """coords_batch_nm: (B, n_atoms, 3) in nm.

        Returns energies_kJmol (B,), forces_kJmol_per_nm (B, n_atoms, 3).
        """
        B = coords_batch_nm.shape[0]
        assert B == self.n_chains, f"expected {self.n_chains} chains, got {B}"
        e = np.empty(B, dtype=np.float64)
        f = np.empty((B, self.n_atoms, 3), dtype=np.float64)
        for i, ctx in enumerate(self.contexts):
            ctx.setPositions(coords_batch_nm[i] * unit.nanometer)
            state = ctx.getState(getEnergy=True, getForces=True)
            e[i] = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            f[i] = np.asarray(state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer))
        return e, f


# ──────────────────────────────────────────────────────────────────────
# OpenMM native Langevin MD — reference trajectory generator
# ──────────────────────────────────────────────────────────────────────

def langevin_sample(system, positions_nm, topology,
                     n_steps: int = 100_000,
                     step_size_fs: float = 2.0,
                     T_kelvin: float = 300.0,
                     friction_ps: float = 1.0,
                     save_every: int = 100,
                     platform_name: str = "OpenCL",
                     seed: int = 0):
    """Run openmm.LangevinIntegrator, return (trajectory_nm, n_steps_saved).

    trajectory_nm: (n_saved, n_atoms, 3)
    """
    integrator = openmm.LangevinIntegrator(
        T_kelvin * unit.kelvin,
        friction_ps / unit.picosecond,
        step_size_fs * unit.femtoseconds,
    )
    integrator.setRandomNumberSeed(seed)
    platform = openmm.Platform.getPlatformByName(platform_name)
    ctx = openmm.Context(system, integrator, platform)
    ctx.setPositions(positions_nm * unit.nanometer)
    ctx.setVelocitiesToTemperature(T_kelvin * unit.kelvin, seed)

    import time
    n_atoms = system.getNumParticles()
    n_save = n_steps // save_every
    out = np.empty((n_save, n_atoms, 3), dtype=np.float64)
    t0 = time.time()
    for i in range(n_save):
        integrator.step(save_every)
        state = ctx.getState(getPositions=True)
        out[i] = np.asarray(state.getPositions(asNumpy=True).value_in_unit(
            unit.nanometer))
    elapsed = time.time() - t0
    return out, elapsed


# ──────────────────────────────────────────────────────────────────────
# Torsion-angle evaluation
# ──────────────────────────────────────────────────────────────────────

def dihedral_np(coords, i, j, k, l):
    """Compute dihedral angle (radians) for 4 atoms (i,j,k,l) on a
    (..., n_atoms, 3) coordinate array.
    """
    b0 = coords[..., j, :] - coords[..., i, :]
    b1 = coords[..., k, :] - coords[..., j, :]
    b2 = coords[..., l, :] - coords[..., k, :]
    b1_norm = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - np.sum(b0 * b1_norm, axis=-1, keepdims=True) * b1_norm
    w = b2 - np.sum(b2 * b1_norm, axis=-1, keepdims=True) * b1_norm
    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1_norm, v) * w, axis=-1)
    return np.arctan2(y, x)


def compute_phi_psi(coords, phi_ij_k_l, psi_ij_k_l):
    """coords: (..., 22, 3) in nm.  Returns (phi, psi) in radians."""
    phi = dihedral_np(coords, *phi_ij_k_l)
    psi = dihedral_np(coords, *psi_ij_k_l)
    return phi, psi


if __name__ == "__main__":
    # Smoke test: build system, compute energy at initial config, print.
    topo, sys_, pos, atoms = build_system()
    print(f"Built alanine dipeptide: {sys_.getNumParticles()} atoms")
    print(f"Atom names: {atoms}")
    phi_ij, psi_ij = phi_psi_atom_indices(topo)
    print(f"φ atoms: {phi_ij}  ψ atoms: {psi_ij}")
    oe = OpenMMEnergy(sys_, n_chains=1)
    e, f = oe.energy_forces(pos[None])
    print(f"Initial energy: {e[0]:.3f} kJ/mol  "
          f"→ {e[0]/kT_kJmol_300K:.3f} kT at 300 K")
    print(f"Max |force| at init: {np.abs(f).max():.2f} kJ/mol/nm")
