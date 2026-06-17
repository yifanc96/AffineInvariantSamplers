"""GNDR (Gauss-Newton Delayed Rejection) on alanine dipeptide via OpenMM.

Residual: harmonic bonds + harmonic angles + isotropic prior  (see
``MD/alanine/residual.py``).

log_prob : −E_total(x)/kT via OpenMM callback (full AMBER + GBSA energy).

Each GNDR iteration:
    1. Evaluate J_r(x), build H_GN = J_rᵀ J_r
    2. Draw proposal with GN-preconditioned Langevin + delayed-rejection
    3. MH-accept using full log_prob (so Coulomb/GBSA mismatch is corrected)
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import ShapeDtypeStruct

import openmm
import openmm.unit as unit

from affine_invariant_samplers import sampler_gndr
from MD.alanine.alanine import (
    build_system, phi_psi_atom_indices, dihedral_np, OpenMMEnergy, kT_kJmol_300K,
)
from MD.alanine.residual import make_alanine_residual
from MD.alanine.run_langevin import ramachandran_kl

ROOT = Path(__file__).resolve().parent


def make_single_point_log_prob(system, kT: float, n_chains: int,
                                  platform_name: str = "OpenCL"):
    """Single-point (D,) → scalar and batched (n, D) → (n,) log_prob.

    Two separate pure_callbacks with distinct output shapes.  The
    sampler's internal jax.vmap over lp_single dispatches n_chains
    serial OpenMM calls (vmap_method="sequential").
    """
    oe1 = OpenMMEnergy(system, n_chains=1, platform_name=platform_name)
    oe = OpenMMEnergy(system, n_chains=n_chains, platform_name=platform_name)
    D = 3 * oe.n_atoms
    inv_kT = 1.0 / kT

    def _batched_host(x_flat):
        x = np.asarray(x_flat, dtype=np.float64).reshape(-1, oe.n_atoms, 3)
        e, f = oe.energy_forces(x)
        return e.astype(np.float64), f.reshape(-1, D).astype(np.float64)

    e_spec_b = ShapeDtypeStruct((n_chains,),    jnp.float64)
    f_spec_b = ShapeDtypeStruct((n_chains, D),  jnp.float64)

    def lp_batched(x_flat):
        e, _ = jax.pure_callback(_batched_host, (e_spec_b, f_spec_b), x_flat,
                                   vmap_method="sequential")
        return -e * inv_kT

    def _single_host(x_flat):
        x = np.asarray(x_flat, dtype=np.float64).reshape(1, oe1.n_atoms, 3)
        e, f = oe1.energy_forces(x)
        return e[0].astype(np.float64), f[0].reshape(-1).astype(np.float64)

    e_spec_s = ShapeDtypeStruct((),    jnp.float64)
    f_spec_s = ShapeDtypeStruct((D,),  jnp.float64)

    @jax.custom_vjp
    def lp_single(x_flat):
        e, _ = jax.pure_callback(_single_host, (e_spec_s, f_spec_s), x_flat,
                                   vmap_method="sequential")
        return -e * inv_kT

    def lp_single_fwd(x_flat):
        e, f = jax.pure_callback(_single_host, (e_spec_s, f_spec_s), x_flat,
                                   vmap_method="sequential")
        return -e * inv_kT, f                                           # residual = force

    def lp_single_bwd(force, g):
        return (g * force * inv_kT,)

    lp_single.defvjp(lp_single_fwd, lp_single_bwd)
    grad_lp_single = jax.grad(lp_single)
    return lp_single, lp_batched, grad_lp_single, oe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-chains",    type=int,   default=16)
    ap.add_argument("--warmup",      type=int,   default=500)
    ap.add_argument("--n-samples",   type=int,   default=3000)
    ap.add_argument("--step-size",   type=float, default=0.05)
    ap.add_argument("--n-try",       type=int,   default=3)
    ap.add_argument("--shrink",      type=float, default=0.3)
    ap.add_argument("--target-accept", type=float, default=0.574)
    ap.add_argument("--sigma-iso",   type=float, default=0.3)
    ap.add_argument("--seed",        type=int,   default=0)
    ap.add_argument("--platform",    default="OpenCL")
    ap.add_argument("--init",        default="gt", choices=["gt", "equilibrium"])
    ap.add_argument("--init-jitter", type=float, default=0.005)
    ap.add_argument("--minimize-init", action="store_true", default=True)
    ap.add_argument("--out", default="gndr")
    ap.add_argument("--gt", default=str(ROOT / "reference" / "AD2_relaxed.npy"))
    args = ap.parse_args()

    out = ROOT / "results" / args.out
    out.mkdir(parents=True, exist_ok=True)

    topo, sys_, pos, atoms = build_system()
    n_atoms = sys_.getNumParticles()
    D = 3 * n_atoms
    phi_ij, psi_ij = phi_psi_atom_indices(topo)

    print(f"[alanine] GNDR  n_chains={args.n_chains}  warmup={args.warmup}  "
          f"n_samples={args.n_samples}  h={args.step_size}  "
          f"n_try={args.n_try}  σ_iso={args.sigma_iso}", flush=True)

    # Build energy callback
    lp_single, lp_batched, grad_lp_single, oe = make_single_point_log_prob(
        sys_, kT_kJmol_300K, n_chains=args.n_chains,
        platform_name=args.platform)

    # Build residual
    residual_single, res_info = make_alanine_residual(
        sys_, pos.flatten(), kT=kT_kJmol_300K, sigma_iso=args.sigma_iso)
    print(f"residual components: bonds={res_info['n_bonds']}  "
          f"angles={res_info['n_angles']}  iso={res_info['n_iso']}  "
          f"total N_res={sum(res_info.values())}", flush=True)

    # Initial ensemble
    gt = np.load(args.gt)                                               # (N_gt, 22, 3) Å
    rng = np.random.default_rng(args.seed)
    if args.init == "gt":
        idx = rng.choice(gt.shape[0], size=args.n_chains, replace=False)
        chains = gt[idx] * 0.1
        chains = chains - chains.mean(axis=1, keepdims=True) + pos.mean(0)
    else:
        chains = np.broadcast_to(pos, (args.n_chains, n_atoms, 3)).copy()
    if args.init_jitter > 0:
        chains = chains + args.init_jitter * rng.normal(size=chains.shape)

    if args.minimize_init:
        plat = openmm.Platform.getPlatformByName("CPU")
        print(f"Minimising {args.n_chains} init configs ...", flush=True)
        t_min = time.time()
        for i in range(args.n_chains):
            integ = openmm.VerletIntegrator(1e-3)
            ctx = openmm.Context(sys_, integ, plat)
            ctx.setPositions(chains[i] * unit.nanometer)
            openmm.LocalEnergyMinimizer.minimize(ctx, tolerance=1.0,
                                                   maxIterations=100)
            chains[i] = np.asarray(ctx.getState(getPositions=True)
                                    .getPositions(asNumpy=True)
                                    .value_in_unit(unit.nanometer))
        print(f"  minimised in {time.time()-t_min:.1f}s", flush=True)

    init_flat = jnp.asarray(chains.reshape(args.n_chains, D))

    # Energies at init
    e0 = np.asarray(lp_batched(init_flat)) * (-kT_kJmol_300K)
    print(f"init energies: min={e0.min():.1f}  max={e0.max():.1f}  "
          f"mean={e0.mean():.1f} kJ/mol", flush=True)

    t0 = time.time()
    samples, info = sampler_gndr(
        lp_single, init_flat, args.n_samples, warmup=args.warmup,
        step_size=args.step_size, n_try=args.n_try, shrink=args.shrink,
        target_accept=args.target_accept,
        residual_fn=residual_single,
        grad_fn=grad_lp_single,
        seed=args.seed, verbose=True,
    )
    samples.block_until_ready()
    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s  acceptance={info['acceptance_rate']:.3f}  "
          f"final_h={info['final_step_size']:.4f}", flush=True)

    arr = np.asarray(samples).reshape(-1, n_atoms, 3)
    finite = np.isfinite(arr).all(axis=(1, 2))
    arr = arr[finite]

    phi_our = dihedral_np(arr, *phi_ij)
    psi_our = dihedral_np(arr, *psi_ij)
    phi_gt  = dihedral_np(gt, *phi_ij)
    psi_gt  = dihedral_np(gt, *psi_ij)

    kl_ours_to_gt = ramachandran_kl(phi_our, psi_our, phi_gt, psi_gt)
    kl_gt_to_ours = ramachandran_kl(phi_gt,  psi_gt,  phi_our, psi_our)
    print(f"KL(ours‖gt) = {kl_ours_to_gt:.3f} bits", flush=True)
    print(f"KL(gt‖ours) = {kl_gt_to_ours:.3f} bits", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    for ax, (p, s, name) in zip(axes,
            [(phi_gt, psi_gt, "GT (AD2_relaxed)"),
             (phi_our, psi_our,
              f"GNDR  {args.n_chains}×{args.n_samples}")]):
        h, xe, ye = np.histogram2d(np.rad2deg(p), np.rad2deg(s),
                                     bins=80, range=[[-180, 180], [-180, 180]])
        ax.pcolormesh(xe, ye, np.log(h.T + 1), cmap="viridis", shading="auto")
        ax.set_title(name)
        ax.set_xlabel("φ (deg)"); ax.set_ylabel("ψ (deg)")
        ax.set_aspect("equal")
    fig.suptitle(f"Ramachandran — KL(ours‖gt)={kl_ours_to_gt:.3f} bits")
    fig.tight_layout()
    fig.savefig(out / "ramachandran.png", dpi=120)

    m = dict(
        sampler="gndr", n_chains=args.n_chains, warmup=args.warmup,
        n_samples=args.n_samples, step_size=args.step_size, n_try=args.n_try,
        shrink=args.shrink, sigma_iso=args.sigma_iso,
        init=args.init, init_jitter=args.init_jitter,
        residual_info=res_info,
        acceptance_rate=float(info["acceptance_rate"]),
        stage1_rate=float(info.get("stage1_rate", -1.0))
                      if "stage1_rate" in info else None,
        final_step_size=float(info["final_step_size"]),
        elapsed_sec=elapsed,
        n_frames=int(arr.shape[0]),
        kl_ours_to_gt_bits=kl_ours_to_gt,
        kl_gt_to_ours_bits=kl_gt_to_ours,
    )
    with open(out / "metrics.json", "w") as f:
        json.dump(m, f, indent=2, default=str)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
