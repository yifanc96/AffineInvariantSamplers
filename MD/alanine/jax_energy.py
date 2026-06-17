"""Wrap OpenMM energy/forces as a JAX-differentiable log-prob.

    log_prob(x) = −E(x) / kT  with x in nm, kT in kJ/mol

Uses ``jax.pure_callback`` for the forward call and ``custom_vjp`` so that
backward uses OpenMM's forces (F = −∇E).
"""
from __future__ import annotations

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import ShapeDtypeStruct

from MD.alanine.alanine import OpenMMEnergy, kT_kJmol_300K


def make_jax_log_prob(system, n_chains: int, kT: float = kT_kJmol_300K,
                       platform_name: str = "OpenCL"):
    """Return a batched log_prob(x: (n_chains, 3N)) → (n_chains,).

    Forward: OpenMM energy via pure_callback.  Backward: uses OpenMM forces
    cached on the forward pass.  Not JIT-traceable — the sampler's outer
    loop stays in Python.
    """
    oe = OpenMMEnergy(system, n_chains=n_chains, platform_name=platform_name)
    n_atoms = oe.n_atoms
    D = 3 * n_atoms
    inv_kT = 1.0 / kT

    def _fwd_host(x_flat):
        coords = np.asarray(x_flat, dtype=np.float64).reshape(-1, n_atoms, 3)
        e, f = oe.energy_forces(coords)
        return e.astype(np.float64), f.reshape(-1, D).astype(np.float64)

    e_spec = ShapeDtypeStruct((n_chains,), jnp.float64)
    f_spec = ShapeDtypeStruct((n_chains, D), jnp.float64)

    @jax.custom_vjp
    def lp_batched(x_flat):
        e, _ = jax.pure_callback(_fwd_host, (e_spec, f_spec), x_flat)
        return -e * inv_kT

    def lp_fwd(x_flat):
        e, f = jax.pure_callback(_fwd_host, (e_spec, f_spec), x_flat)
        return -e * inv_kT, f

    def lp_bwd(force, g):
        return (g[:, None] * force * inv_kT,)

    lp_batched.defvjp(lp_fwd, lp_bwd)
    return lp_batched, oe
