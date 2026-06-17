"""pickles_unadjusted on alanine dipeptide via OpenMM energy callback.

    N parallel chains in JAX land, each energy eval dispatches to OpenMM
    (OpenCL) through `jax.pure_callback`.  BAOAB Langevin integrator with
    affine-invariant ensemble preconditioning.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax, jax.numpy as jnp

from affine_invariant_samplers import sampler_pickles_unadjusted
from MD.alanine.alanine import (build_system, phi_psi_atom_indices,
                                  dihedral_np, kT_kJmol_300K)
from MD.alanine.jax_energy import make_jax_log_prob
from MD.alanine.run_langevin import ramachandran_kl

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-chains",    type=int,   default=64)
    ap.add_argument("--warmup",      type=int,   default=500)
    ap.add_argument("--n-samples",   type=int,   default=2000)
    ap.add_argument("--step-size",   type=float, default=0.002)
    ap.add_argument("--gamma",       type=float, default=2.0)
    ap.add_argument("--adapt",       default="none",
                    choices=["none", "maxdrift", "pgn", "samadams"],
                    help="step-size adapter (default: none — alanine has "
                         "large grad norms, adapter collapses h)")
    ap.add_argument("--target-gnorm", type=float, default=100.0,
                    help="only used when --adapt != none")
    ap.add_argument("--init-jitter", type=float, default=0.01,
                    help="Gaussian noise (nm) added to initial positions")
    ap.add_argument("--init",        default="gt",
                    choices=["gt", "equilibrium"])
    ap.add_argument("--minimize-init", action="store_true", default=True,
                    help="Run OpenMM L-BFGS minimizer on each init config "
                         "(needed when GT was generated with a different FF)")
    ap.add_argument("--no-minimize-init", dest="minimize_init",
                    action="store_false")
    ap.add_argument("--thin-by",     type=int,   default=1)
    ap.add_argument("--seed",        type=int,   default=0)
    ap.add_argument("--platform",    default="OpenCL")
    ap.add_argument("--out",         default="pickles_unadjusted")
    ap.add_argument("--gt",          default=str(ROOT / "reference" / "AD2_relaxed.npy"))
    args = ap.parse_args()

    out = ROOT / "results" / args.out
    out.mkdir(parents=True, exist_ok=True)

    topo, sys_, pos, atoms = build_system()
    n_atoms = sys_.getNumParticles()
    D = 3 * n_atoms
    phi_ij, psi_ij = phi_psi_atom_indices(topo)

    print(f"Building JAX→OpenMM callback ({args.n_chains} chains, "
          f"{args.platform})...", flush=True)
    lp_batched, oe = make_jax_log_prob(sys_, n_chains=args.n_chains,
                                         platform_name=args.platform)

    # Explicit batched-gradient function avoids the default vmap(grad)
    # path (which would break our pure_callback: pure_callback expects
    # the full batch of n_chains).
    @jax.jit
    def grad_lp_batched(x_flat):
        return jax.grad(lambda xx: lp_batched(xx).sum())(x_flat)

    # Initial ensemble
    gt = np.load(args.gt)                                               # (N_gt, 22, 3) in Å
    rng = np.random.default_rng(args.seed)
    if args.init == "gt":
        # Convert Å → nm.  Also re-centre each frame so the COM matches
        # the bgmol reference (since the absolute translation is arbitrary).
        idx = rng.choice(gt.shape[0], size=args.n_chains, replace=False)
        chains = gt[idx] * 0.1                                           # → nm
        # Re-centre to the bgmol equilibrium COM so the implicit-solvent
        # GB-OBC neighbour lookup starts from a reasonable place.
        chains = chains - chains.mean(axis=1, keepdims=True) + pos.mean(0)
    else:
        chains = np.broadcast_to(pos, (args.n_chains, n_atoms, 3)).copy()

    if args.init_jitter > 0:
        chains = chains + args.init_jitter * rng.normal(size=chains.shape)

    if args.minimize_init:
        import openmm, openmm.unit as unit
        plat = openmm.Platform.getPlatformByName("CPU")
        print(f"Minimising {args.n_chains} init configs...", flush=True)
        t_min = time.time()
        for i in range(args.n_chains):
            integ = openmm.VerletIntegrator(1e-3)
            ctx = openmm.Context(sys_, integ, plat)
            ctx.setPositions(chains[i] * unit.nanometer)
            openmm.LocalEnergyMinimizer.minimize(ctx, tolerance=1.0,
                                                   maxIterations=100)
            state = ctx.getState(getPositions=True)
            chains[i] = np.asarray(state.getPositions(asNumpy=True)
                                    .value_in_unit(unit.nanometer))
        print(f"Minimised in {time.time() - t_min:.1f}s", flush=True)

    init_flat = jnp.asarray(chains.reshape(args.n_chains, D), dtype=jnp.float64)

    # Check the energy at init
    e0 = np.asarray(lp_batched(init_flat)) * (-kT_kJmol_300K)
    print(f"Init energies: min={e0.min():.1f}  max={e0.max():.1f}  "
          f"mean={e0.mean():.1f} kJ/mol  ({e0.mean()/kT_kJmol_300K:+.1f} kT)",
          flush=True)

    print(f"[alanine] pickles_unadjusted  n_chains={args.n_chains}  "
          f"warmup={args.warmup}  n_samples={args.n_samples}  "
          f"h={args.step_size}  γ={args.gamma}  thin_by={args.thin_by}",
          flush=True)

    t0 = time.time()
    samples, info = sampler_pickles_unadjusted(
        lp_batched, init_flat, args.n_samples,
        step_size=args.step_size, gamma=args.gamma,
        warmup=args.warmup, thin_by=args.thin_by,
        adapt=None if args.adapt == "none" else args.adapt,
        target_gnorm=args.target_gnorm,
        grad_log_prob_fn=grad_lp_batched,
        seed=args.seed, verbose=True)
    samples.block_until_ready()
    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s", flush=True)

    arr = np.asarray(samples).reshape(-1, n_atoms, 3)                   # (S*N, 22, 3)
    finite = np.isfinite(arr).all(axis=(1, 2))
    arr = arr[finite]
    print(f"{arr.shape[0]} total samples ({finite.mean():.3f} finite)",
          flush=True)

    # torsions
    phi_our = dihedral_np(arr, *phi_ij)
    psi_our = dihedral_np(arr, *psi_ij)
    phi_gt  = dihedral_np(gt, *phi_ij)
    psi_gt  = dihedral_np(gt, *psi_ij)

    kl_ours_to_gt = ramachandran_kl(phi_our, psi_our, phi_gt, psi_gt)
    kl_gt_to_ours = ramachandran_kl(phi_gt,  psi_gt,  phi_our, psi_our)
    print(f"KL(ours‖gt) = {kl_ours_to_gt:.3f} bits", flush=True)
    print(f"KL(gt‖ours) = {kl_gt_to_ours:.3f} bits", flush=True)

    # Ramachandran figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    for ax, (p, s, name) in zip(axes,
                                  [(phi_gt, psi_gt, "GT (AD2_relaxed)"),
                                   (phi_our, psi_our,
                                    f"pickles_unadj {args.n_samples}×{args.n_chains}")]):
        h, xe, ye = np.histogram2d(np.rad2deg(p), np.rad2deg(s),
                                     bins=80,
                                     range=[[-180, 180], [-180, 180]])
        ax.pcolormesh(xe, ye, np.log(h.T + 1), cmap="viridis", shading="auto")
        ax.set_title(name)
        ax.set_xlabel("φ (deg)"); ax.set_ylabel("ψ (deg)")
        ax.set_aspect("equal")
    fig.suptitle(f"Ramachandran — KL(ours‖gt)={kl_ours_to_gt:.3f} bits")
    fig.tight_layout()
    fig.savefig(out / "ramachandran.png", dpi=120)

    m = dict(
        sampler="pickles_unadjusted",
        n_chains=args.n_chains, warmup=args.warmup, n_samples=args.n_samples,
        step_size=args.step_size, gamma=args.gamma, thin_by=args.thin_by,
        init=args.init, init_jitter=args.init_jitter,
        n_frames_total=int(arr.shape[0]),
        finite_frac=float(finite.mean()),
        final_step_size=float(info.get("step_size", args.step_size))
                          if hasattr(info, "get") else args.step_size,
        elapsed_sec=elapsed,
        kl_ours_to_gt_bits=kl_ours_to_gt,
        kl_gt_to_ours_bits=kl_gt_to_ours,
    )
    with open(out / "metrics.json", "w") as f:
        json.dump(m, f, indent=2, default=str)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
