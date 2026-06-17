"""Reference MD run for alanine dipeptide — openmm.LangevinIntegrator.

    20 ns at 2 fs step at T=300 K (with ff99SBildn + GB-OBC).
    Compares φ, ψ density vs AD2_relaxed.npy.

Outputs:
    results/langevin/trajectory.npy      (n_saved, 22, 3) in nm
    results/langevin/metrics.json        phi/psi KL, wall time
    results/langevin/ramachandran.png    side-by-side plot
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from MD.alanine.alanine import (
    build_system, phi_psi_atom_indices, dihedral_np, langevin_sample,
    kT_kJmol_300K, OpenMMEnergy,
)

ROOT = Path(__file__).resolve().parent


def ramachandran_kl(phi_a, psi_a, phi_b, psi_b, bins: int = 60):
    """KL(p_a || p_b) on a regular 2-D grid over [-π,π]² (bits)."""
    edges = np.linspace(-np.pi, np.pi, bins + 1)
    ha, _, _ = np.histogram2d(phi_a, psi_a, bins=[edges, edges])
    hb, _, _ = np.histogram2d(phi_b, psi_b, bins=[edges, edges])
    eps = 1e-8
    pa = ha / ha.sum() + eps
    pb = hb / hb.sum() + eps
    pa /= pa.sum(); pb /= pb.sum()
    return float(np.sum(pa * (np.log2(pa) - np.log2(pb))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-steps", type=int, default=10_000_000,
                    help="# MD steps (2 fs each; 10M = 20 ns)")
    ap.add_argument("--save-every", type=int, default=500,
                    help="save a frame every N steps")
    ap.add_argument("--T", type=float, default=300.0)
    ap.add_argument("--platform", default="OpenCL")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="langevin")
    ap.add_argument("--gt", default=str(ROOT / "reference" / "AD2_relaxed.npy"))
    args = ap.parse_args()

    out = ROOT / "results" / args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"Building system...", flush=True)
    topo, sys_, pos, atoms = build_system()
    phi_ij, psi_ij = phi_psi_atom_indices(topo)
    print(f"  {sys_.getNumParticles()} atoms  φ={phi_ij}  ψ={psi_ij}", flush=True)

    print(f"Running Langevin: {args.n_steps:,} steps × 2 fs "
          f"= {args.n_steps * 2e-6:.1f} ns at T={args.T} K", flush=True)
    t0 = time.time()
    traj, md_wall = langevin_sample(
        sys_, pos, topo,
        n_steps=args.n_steps, save_every=args.save_every,
        T_kelvin=args.T, seed=args.seed,
        platform_name=args.platform,
    )
    print(f"MD done in {md_wall:.1f}s  ({args.n_steps / md_wall:.0f} steps/s) "
          f"→ {traj.shape[0]} saved frames", flush=True)
    np.save(out / "trajectory.npy", traj)

    phi_our = dihedral_np(traj, *phi_ij)
    psi_our = dihedral_np(traj, *psi_ij)

    gt = np.load(args.gt)
    phi_gt = dihedral_np(gt, *phi_ij)
    psi_gt = dihedral_np(gt, *psi_ij)

    kl_ours_to_gt = ramachandran_kl(phi_our, psi_our, phi_gt, psi_gt)
    kl_gt_to_ours = ramachandran_kl(phi_gt, psi_gt, phi_our, psi_our)
    print(f"Ramachandran KL(ours‖gt) = {kl_ours_to_gt:.3f} bits", flush=True)
    print(f"Ramachandran KL(gt‖ours) = {kl_gt_to_ours:.3f} bits", flush=True)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    for ax, (phi, psi, name) in zip(
            axes, [(phi_gt, psi_gt, "GT (AD2_relaxed)"),
                   (phi_our, psi_our, f"Langevin {args.n_steps*2e-6:.0f} ns")]):
        h, xe, ye = np.histogram2d(np.rad2deg(phi), np.rad2deg(psi),
                                     bins=80,
                                     range=[[-180, 180], [-180, 180]])
        ax.pcolormesh(xe, ye, np.log(h.T + 1), cmap="viridis", shading="auto")
        ax.set_title(name)
        ax.set_xlabel("φ (deg)"); ax.set_ylabel("ψ (deg)")
        ax.set_aspect("equal")
    fig.suptitle(f"Ramachandran — KL(ours‖gt)={kl_ours_to_gt:.3f} bits")
    fig.tight_layout()
    fig.savefig(out / "ramachandran.png", dpi=120)
    print(f"figure: {out / 'ramachandran.png'}")

    # Metrics
    m = dict(
        sampler="openmm_langevin",
        n_steps=args.n_steps, save_every=args.save_every,
        T=args.T, seed=args.seed, platform=args.platform,
        md_sec=md_wall, total_sec=time.time() - t0,
        n_frames=int(traj.shape[0]),
        kl_ours_to_gt_bits=kl_ours_to_gt,
        kl_gt_to_ours_bits=kl_gt_to_ours,
    )
    with open(out / "metrics.json", "w") as f:
        json.dump(m, f, indent=2)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
